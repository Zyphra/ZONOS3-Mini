#!/usr/bin/env bash
# start-zonos3 — launch zonos3-server from local GGUF files; fetch ffmpeg if needed.
# Double-click friendly: on macOS this file also ships as start-zonos3.command.
# Must stay bash-3.2 compatible (stock macOS bash).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_DEFAULT=cpu   # CI stamps this to "gpu" in the vulkan/metal release archives
ZONOS3_QUANT="${ZONOS3_QUANT:-ptq}"
ZONOS3_ZAC="${ZONOS3_ZAC:-zac-keep2}"
ZONOS3_MODEL_DIR="${ZONOS3_MODEL_DIR:-$SCRIPT_DIR/models}"
ZONOS3_HOST="${ZONOS3_HOST:-127.0.0.1}"
ZONOS3_PORT="${ZONOS3_PORT:-1919}"
ZONOS3_SERVER_BIN="${ZONOS3_SERVER_BIN:-$SCRIPT_DIR/zonos3-server}"
ASSUME_YES="${ZONOS3_ASSUME_YES:-0}"
NO_BROWSER="${ZONOS3_NO_BROWSER:-0}"
FFMPEG_URL="${ZONOS3_FFMPEG_URL:-}"   # override with a single-binary URL (used by tests); default is per-OS
FFMPEG_DIR="$ZONOS3_MODEL_DIR/bin"
FFMPEG_BIN="$FFMPEG_DIR/ffmpeg"
GPU_MODE="$GPU_DEFAULT"
EXTRA_ARGS=()

usage() {
    cat <<EOF
usage: $0 [options] [-- <extra zonos3-server args>]
  --quant Q      backbone file models/zonos3-Q.gguf (default ptq; qat = the previous mect_60k backbone)
  --zac Z        decoder file models/Z.gguf (default zac-keep2, the keep2 decoder; zac = the full-size one)
  --cpu | --gpu  backend (archive default: $GPU_DEFAULT)
  --host H       bind address (default $ZONOS3_HOST)
  --port P       port (default $ZONOS3_PORT)
  -y, --yes      don't ask before fetching ffmpeg
  --no-browser   don't open the web UI
env overrides: ZONOS3_QUANT ZONOS3_ZAC ZONOS3_MODEL_DIR ZONOS3_HOST ZONOS3_PORT
               ZONOS3_SERVER_BIN ZONOS3_ASSUME_YES ZONOS3_NO_BROWSER ZONOS3_FFMPEG_URL
ffmpeg (voice cloning only) is fetched into $ZONOS3_MODEL_DIR/bin if not already there or
on PATH; basic TTS needs no ffmpeg.
EOF
}

die() { echo "start-zonos3: error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --quant)      [ $# -ge 2 ] || die "--quant needs a value"; ZONOS3_QUANT="$2"; shift 2 ;;
        --zac)        [ $# -ge 2 ] || die "--zac needs a value"; ZONOS3_ZAC="$2"; shift 2 ;;
        --cpu)        GPU_MODE=cpu; shift ;;
        --gpu)        GPU_MODE=gpu; shift ;;
        --host)       [ $# -ge 2 ] || die "--host needs a value"; ZONOS3_HOST="$2"; shift 2 ;;
        --port)       [ $# -ge 2 ] || die "--port needs a value"; ZONOS3_PORT="$2"; shift 2 ;;
        -y|--yes)     ASSUME_YES=1; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA_ARGS=("$@"); break ;;
        *)            usage; die "unknown option '$1' (pass server flags after --)" ;;
    esac
done


ffmpeg_size_mb() { case "$(uname -s)" in Darwin) echo 45 ;; *) echo 108 ;; esac; }

# Fetch a static ffmpeg into $FFMPEG_DIR — only needed for voice cloning (decoding the
# reference audio); basic TTS never touches it. Linux uses the BtbN LGPL build; macOS a
# pinned static arm64 binary (GPL, but we don't redistribute it). Override with
# ZONOS3_FFMPEG_URL.
ffmpeg_download() {
    mkdir -p "$FFMPEG_DIR" || return 1
    ff_url=""; ff_single=1
    if [ -n "$FFMPEG_URL" ]; then
        ff_url="$FFMPEG_URL"
    elif [ "$(uname -s)" = Darwin ]; then
        ff_url="https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64"
    else
        ff_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-7.1.tar.xz"
        ff_single=0
    fi
    echo "downloading ffmpeg ($(ffmpeg_size_mb) MB, for voice cloning) ..."
    if [ "$ff_single" = 1 ]; then
        curl -L --fail --retry 3 --progress-bar -o "$FFMPEG_BIN.part" "$ff_url" || return 1
        chmod +x "$FFMPEG_BIN.part"; mv "$FFMPEG_BIN.part" "$FFMPEG_BIN"
    else
        # BtbN archive nests the binary at ffmpeg-*/bin/ffmpeg; extract, relocate, clean up
        ff_tmp="$FFMPEG_DIR/ffmpeg-dl.tar.xz"; ff_x="$FFMPEG_DIR/.extract"
        rm -rf "$ff_x"; mkdir -p "$ff_x"
        curl -L --fail --retry 3 --progress-bar -o "$ff_tmp" "$ff_url" || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        tar -xf "$ff_tmp" -C "$ff_x" || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        ff_found=$(find "$ff_x" -type f -name ffmpeg -path '*/bin/*' 2>/dev/null | head -1)
        [ -n "$ff_found" ] || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        mv "$ff_found" "$FFMPEG_BIN"; chmod +x "$FFMPEG_BIN"
        rm -rf "$ff_tmp" "$ff_x"
    fi
    [ -x "$FFMPEG_BIN" ]
}


[ -x "$ZONOS3_SERVER_BIN" ] || die "zonos3-server not found at $ZONOS3_SERVER_BIN (run this script from the extracted release archive, or set ZONOS3_SERVER_BIN)"

BACKBONE="zonos3-$ZONOS3_QUANT.gguf"
DECODER="$ZONOS3_ZAC.gguf"
MISSING=()
for f in "$BACKBONE" "$DECODER"; do
    [ -f "$ZONOS3_MODEL_DIR/$f" ] || MISSING+=("$f")
done
[ "${#MISSING[@]}" -eq 0 ] ||
    die "missing ${MISSING[*]} in $ZONOS3_MODEL_DIR — create them with models/convert-*.py (docs/INTERNALS.md)"
[ -f "$ZONOS3_MODEL_DIR/spk-encoder.gguf" ] ||
    echo "note: spk-encoder.gguf missing — speaker-embedding cloning disabled (raw-reference cloning needs zac-encoder.gguf + ssl-teacher.gguf)" >&2
for companion in zac-encoder.gguf ssl-teacher.gguf; do
    [ -f "$ZONOS3_MODEL_DIR/$companion" ] || echo "note: $companion not found — raw-reference cloning needs both zac-encoder.gguf and ssl-teacher.gguf" >&2
done

# ffmpeg is only needed for voice cloning. Prefer our own copy, then PATH.
NEED_FFMPEG=0
if [ -x "$FFMPEG_BIN" ]; then
    export ZONOS3_FFMPEG="$FFMPEG_BIN"
elif [ "${ZONOS3_SKIP_PATH_FFMPEG:-0}" != 1 ] && command -v ffmpeg >/dev/null 2>&1; then
    :   # a system ffmpeg is on PATH; the server/CLI resolve it there (ZONOS3_SKIP_PATH_FFMPEG=1 forces a download, for tests)
else
    NEED_FFMPEG=1
fi

if [ "$NEED_FFMPEG" = 1 ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to fetch ffmpeg"
    echo "ffmpeg  (~$(ffmpeg_size_mb) MB, voice cloning only) is not installed"
    if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
        printf "Fetch ffmpeg now? [Y/n] "
        read -r ans
        case "$ans" in ""|y*|Y*) ;; *) echo "aborted."; exit 1 ;; esac
    fi
    if ffmpeg_download; then
        export ZONOS3_FFMPEG="$FFMPEG_BIN"
    else
        echo "start-zonos3: warning: ffmpeg download failed — voice cloning disabled (basic TTS still works)" >&2
    fi
fi

# The default voices ship next to this script in release archives; in a source checkout the script lives in
# servers/ggml/scripts/ and they sit at the repo root. Pass them explicitly because the server default is
# cwd-relative and a double-clicked .command runs from $HOME. The server finds the studio itself (beside the
# executable in an archive, studio/dist in a checkout).
sibling() {  # $1 = name: echo the archive-layout path, falling back to the repo layout
    if [ -e "$SCRIPT_DIR/$1" ]; then echo "$SCRIPT_DIR/$1"; else echo "$SCRIPT_DIR/../../../$1"; fi
}

args=("$ZONOS3_MODEL_DIR/$BACKBONE"
      --zac "$ZONOS3_MODEL_DIR/$DECODER"
      --host "$ZONOS3_HOST" --port "$ZONOS3_PORT")
[ -d "$(sibling default_voices)" ]     && args+=(--tts-default-voices-dir "$(sibling default_voices)")
[ -f "$ZONOS3_MODEL_DIR/zac-encoder.gguf" ] && args+=(--zac-encoder "$ZONOS3_MODEL_DIR/zac-encoder.gguf")
[ -f "$ZONOS3_MODEL_DIR/ssl-teacher.gguf" ] && args+=(--ssl-teacher "$ZONOS3_MODEL_DIR/ssl-teacher.gguf")
[ -f "$ZONOS3_MODEL_DIR/spk-encoder.gguf" ] && args+=(--spk-encoder "$ZONOS3_MODEL_DIR/spk-encoder.gguf")
[ "$GPU_MODE" = gpu ] && args+=(--gpu)

"$ZONOS3_SERVER_BIN" "${args[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' INT TERM EXIT

# Poll readiness; bail out early if the server died (its error is already on the terminal).
POLL_HOST="$ZONOS3_HOST"
case "$POLL_HOST" in 0.0.0.0|::) POLL_HOST=127.0.0.1 ;; esac
URL="http://$POLL_HOST:$ZONOS3_PORT/"
ready=0
i=0
while [ $i -lt 240 ]; do
    kill -0 "$SRV_PID" 2>/dev/null || die "server exited during startup"
    if curl -fsS "${URL}v2/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.5
    i=$((i + 1))
done
[ "$ready" = 1 ] || die "server did not become ready at $URL"

echo "zonos3: ready -> $URL"
if [ "$NO_BROWSER" != 1 ]; then
    case "$(uname -s)" in
        Darwin) open "$URL" 2>/dev/null || true ;;
        *)      command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" 2>/dev/null || true ;;
    esac
fi

echo "Press Ctrl-C (or close this window) to stop the server."
wait "$SRV_PID"
