"""Stage and deploy the ZONOS3 Mini ZeroGPU Space.

    uv run --locked python servers/space/deploy.py [--repo-id Zyphra/ZONOS3-Mini] [--stage-only DIR]

Deploying creates the private Space, stores the local HF login token as its HF_TOKEN secret, requests
ZeroGPU hardware and uploads app.py + the space card + the zonos3 package, steering/ and default_voices/ (from
assets/), FuseAlign's fusealign_hub package, the studio's build (studio/dist, without the in-browser engine) as
studio/ and the Rust text normalizer's wheel (built for the Space's manylinux x86_64 CPython 3.12 in Docker, with
maturin's official image).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware, get_token

SPACE_ID = "Zyphra/ZONOS3-Mini"
REPO = Path(__file__).resolve().parents[2]
SPACE_FILES = ("app.py", "README.md", "requirements.txt", "packages.txt")
# Tracked repo paths and where the Space has them (app.py reads them relative to itself).
TRACKED = {"model/src/zonos3": "zonos3", "assets/steering": "steering", "assets/default_voices": "default_voices"}
# FuseAlign (speech editing's forced aligner), a submodule: its package as tracked there, with its license.
FUSEALIGN = REPO / "third_party" / "fusealign-multilingual"
# The text normalizer (a submodule): its wheel, which requirements.txt installs by this file name. Built in maturin's
# manylinux image (an old glibc) with a container-private cargo target dir: host build scripts in the submodule's
# target/ would not run there.
TEXTNORM = REPO / "third_party" / "zyphra-textnorm"
TEXTNORM_WHEEL = "zyphra_textnorm-0.1.0-cp39-abi3-manylinux_2_28_x86_64.whl"
MATURIN_IMAGE = "ghcr.io/pyo3/maturin:v1.16.0"


def git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def build_textnorm_wheel() -> Path:
    """maturin build of third_party/zyphra-textnorm for linux/amd64 (manylinux_2_28, abi3), into its target/ (under
    the repo: Docker Desktop and colima share only the home directory with containers)."""
    out = TEXTNORM / "target" / "space-wheel"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-e",
            "CARGO_TARGET_DIR=/tmp/target",
            "-v",
            f"{TEXTNORM}:/src",
            "-w",
            "/src",
            MATURIN_IMAGE,
            "build",
            "--release",
            "--locked",
            "--compatibility",
            "manylinux_2_28",
            "-o",
            "/src/target/space-wheel",
        ],
        check=True,
    )
    wheel = out / TEXTNORM_WHEEL
    if not wheel.is_file():
        sys.exit(f"maturin did not produce {TEXTNORM_WHEEL}: {sorted(p.name for p in out.iterdir())}")
    return wheel


def stage(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build_textnorm_wheel(), dest / TEXTNORM_WHEEL)
    for name in SPACE_FILES:
        shutil.copy2(REPO / "servers" / "space" / name, dest / name)
    for path in filter(None, git("ls-files", "-z", "--", *TRACKED).split("\0")):
        source = next(s for s in TRACKED if path.startswith(f"{s}/"))
        target = dest / TRACKED[source] / path.removeprefix(f"{source}/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / path, target)
    for path in filter(None, git("ls-files", "-z", "--", "fusealign_hub", repo=FUSEALIGN).split("\0")):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FUSEALIGN / path, target)
    shutil.copy2(FUSEALIGN / "LICENSE", dest / "fusealign_hub" / "LICENSE")
    # The build as it is on disk (a fresh `pnpm --dir studio build` has untracked hashed assets until committed).
    shutil.copytree(REPO / "studio" / "dist", dest / "studio", ignore=shutil.ignore_patterns("web-engine"))


def deploy(repo_id: str) -> None:
    token = get_token()
    if token is None:
        sys.exit("run 'hf auth login' first")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.add_space_secret(repo_id, "HF_TOKEN", token)
    api.request_space_hardware(repo_id, SpaceHardware.ZERO_A10G)
    describe = git("describe", "--always", "--dirty").strip()
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy ZONOS3 {describe}",
            delete_patterns=["zonos3/**", "steering/**", "default_voices/**", "studio/**", "fusealign_hub/**", "*.whl"],
        )
    print(f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=SPACE_ID)
    parser.add_argument("--stage-only", type=Path, metavar="DIR")
    args = parser.parse_args()
    if args.stage_only is not None:
        if args.stage_only.exists() and any(args.stage_only.iterdir()):
            sys.exit(f"refusing to stage into non-empty {args.stage_only}")
        stage(args.stage_only)
        print(args.stage_only)
        return
    deploy(args.repo_id)


if __name__ == "__main__":
    main()
