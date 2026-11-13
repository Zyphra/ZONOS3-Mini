#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
if [ "$#" -ne 1 ]; then echo 'usage: bash scripts/gen_trace_corpus.sh {kld|calib}' >&2; exit 2; fi
case "$1" in
  kld)
    OUT=out/kldcorp MAN=out/kld.manifest SEED0=1000
    texts=(
      "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains."
      "Why do we dream? Neuroscientists have debated this question for over a century without consensus."
      "In nineteen eighty-five, researchers discovered a new species of deep-sea creature near the vents."
      "Please remember to bring your umbrella, your charger, your passport, and the blue folder tomorrow."
      "What a beautiful morning! The birds are singing and the air smells fresh after last night's storm."
      "Our quarterly revenue grew by twelve percent, driven largely by strong international demand."
      "She whispered softly, but her words carried across the empty concert hall like a haunting melody."
      "Turn left at the second traffic light, continue for half a mile, and the museum is on your right."
    ) ;;
  calib)
    OUT=out/gencal MAN=out/calib.manifest SEED0=5000
    texts=(
      "Economic forecasts suggest that inflation may continue to rise throughout the coming fiscal quarter."
      "Once upon a time, in a kingdom far beyond the northern seas, there lived a curious young inventor."
      "The chemical reaction produced an unexpected burst of violet light and a faint smell of citrus."
      "Breaking news: the championship match has been postponed due to severe weather across the region."
      "He counted slowly from one to twenty, then opened his eyes and began to search the quiet garden."
      "Wait! Don't touch that wire, it's still connected to the main power supply and could be dangerous."
      "I absolutely loved the film, though the ending left me with so many unanswered questions."
      "The recipe calls for two cups of flour, three eggs, a pinch of salt, and a teaspoon of vanilla."
      "Ladies and gentlemen, thank you for joining us this evening for a very special performance."
      "Honestly, I can't believe it's already June; this year has flown by faster than I expected."
      "The spacecraft entered orbit at exactly four thirty in the morning, transmitting data back home."
      "Could you please repeat the address? I want to make absolutely sure I wrote it down correctly."
      "Thunder rolled across the valley as the first heavy drops of rain began to strike the rooftops."
      "He paused at the doorway, took a deep breath, and finally stepped into the crowded interview room."
      "The ancient manuscript described a hidden chamber beneath the temple, guarded by a stone serpent."
      "I'm so sorry for the delay; the train was held up because of a signal failure outside the station."
      "Mathematics is the language in which the universe writes its most elegant and enduring secrets."
      "The toddler giggled with delight as the puppy chased its own tail around and around the living room."
      "According to the report, global temperatures have risen by roughly one degree over the last century."
      "Grab your coat, lock the door, and meet me by the old oak tree at the bottom of the hill at noon."
      "The orchestra tuned their instruments while the audience settled into their plush velvet seats."
      "Remember, courage is not the absence of fear, but the decision that something else matters more."
    ) ;;
  *) echo 'usage: bash scripts/gen_trace_corpus.sh {kld|calib}' >&2; exit 2 ;;
esac
BUILD=${BUILD:-build-cuda}
MODEL=${MODEL:-out/backbone-f16.gguf}
ZAC=${ZAC:-out/zac.gguf}
ZAC_ENCODER=${ZAC_ENCODER:-out/zac-encoder.gguf}
SSL_TEACHER=${SSL_TEACHER:-out/ssl-teacher.gguf}
TRACE_MAX=${TRACE_MAX:-300}
BACKEND=${BACKEND:---gpu}
case "$BACKEND" in --gpu|--cpu) ;; *) echo 'gen_trace_corpus: BACKEND must be --gpu or --cpu' >&2; exit 2 ;; esac
for input in "$BUILD/zonos3-cli" "$BUILD/zac-encoder-cli" "$MODEL" "$ZAC" "$ZAC_ENCODER" "$SSL_TEACHER"; do
  if [ ! -f "$input" ]; then echo "gen_trace_corpus: missing $input" >&2; exit 1; fi
done
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$MAN"
trap 'echo "gen_trace_corpus: FAILED, see $OUT/log.txt" >&2' ERR
run() { printf '%q ' "$@" >> "$OUT/log.txt"; printf '\n' >> "$OUT/log.txt"; "$@" >> "$OUT/log.txt" 2>&1; }
references=()
for v in ../../assets/default_voices/*.mp3; do
  [ -f "$v" ] || continue
  reference="$OUT/reference_$(basename "$v" .mp3).npy"
  args=("$BUILD/zac-encoder-cli" "$ZAC_ENCODER" "$SSL_TEACHER" --audio "$v" "$reference")
  if [ "$BACKEND" = --gpu ]; then args+=(--gpu); fi
  run "${args[@]}"
  references+=("$reference")
done
idx=0
for text in "${texts[@]}"; do
  for reference in '' "${references[@]}"; do
    base="$OUT/seq_$(printf '%03d' "$idx")"
    args=("$BUILD/zonos3-cli" "$MODEL" --tts "$text" "$base.codes.npy" --dump-ids "$base.npy" --zac "$ZAC" --seed "$((SEED0+idx))" --max "$TRACE_MAX" "$BACKEND")
    if [ -n "$reference" ]; then args+=(--reference-codes "$reference"); fi
    run "${args[@]}"
    echo "$base.npy" >> "$MAN"
    idx=$((idx+1))
  done
done
echo "gen_trace_corpus $1: $idx traces (${#references[@]} raw reference voices plus unconditioned prompts) -> $MAN"
