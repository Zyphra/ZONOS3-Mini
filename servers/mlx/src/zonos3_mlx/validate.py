"""One full LM forward with every capture, for tools/compare_golden.py against tools/dump_golden.py lm.

    python -m zonos3_mlx.validate MODEL.gguf IDS.npy OUTDIR

IDS.npy is float32 [rows, n_codebooks + 1] with its ``IDS.cond.json`` sidecar (speaker_row / speaker_embedding,
condition_row / condition_buckets), as dump_golden.py writes them. When ``<IDS dir>/manifest.json`` exists, every name in
its ``expected_tensors`` must be produced; the final cache state (gdn_state_L, gdn_conv_state_L, kv_k_L, kv_v_L) comes
from the LMState after the forward, in dump_golden.py ``save_lm_state`` shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _cond_path(ids_path: Path) -> Path:
    return ids_path.with_suffix(".cond.json") if ids_path.suffix == ".npy" else Path(str(ids_path) + ".cond.json")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    model_path, ids_path, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    from zonos3_mlx.models.lm import LMState, Zonos3MLX

    lm = Zonos3MLX.load(model_path)
    c = lm.config
    ids = np.load(ids_path).astype(np.int64)
    if ids.ndim != 2 or ids.shape[1] != c.n_codebooks + 1:
        print(f"error: {ids_path} must be [rows, {c.n_codebooks + 1}]", file=sys.stderr)
        return 1
    speaker_rows = condition_rows = None
    cond_path = _cond_path(ids_path)
    if cond_path.is_file():
        cond = json.loads(cond_path.read_text())
        if int(cond.get("speaker_row", -1)) >= 0:
            speaker_rows = {int(cond["speaker_row"]): np.asarray(cond["speaker_embedding"], dtype=np.float32)}
        if int(cond.get("condition_row", -1)) >= 0:
            condition_rows = {int(cond["condition_row"]) + i: int(b) for i, b in enumerate(cond["condition_buckets"])}
    capture: dict = {}
    state = LMState()
    lm.forward(ids, state, speaker_rows=speaker_rows, condition_rows=condition_rows, capture=capture)
    for layer in range(c.n_layers):
        if c.uses_gdn(layer):
            capture[f"gdn_state_{layer}"] = np.array(state.gdn_state[layer])
            capture[f"gdn_conv_state_{layer}"] = np.array(state.conv_state[layer])
        else:
            capture[f"kv_k_{layer}"], capture[f"kv_v_{layer}"] = state.kv_rows(layer)
    manifest = ids_path.parent / "manifest.json"
    if manifest.is_file():
        for name in json.loads(manifest.read_text()).get("expected_tensors", []):
            if name not in capture:
                print(f"error: validate cannot produce capture {name}", file=sys.stderr)
                return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, value in capture.items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(value, dtype=np.float32))
    print(f"wrote {len(capture)} captures for {len(ids)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
