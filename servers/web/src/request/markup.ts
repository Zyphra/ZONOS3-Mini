// Nonverbal markup: port of servers/ggml/src/zonos3-markup.cpp (model/src/zonos3/tts/markup.py). Markers match
// </?(?:tag|...)\s*/?> ; offsets are Unicode code points (Python str indices).

export const NONVERBAL_MARKUP_TAGS = ['laugh', 'breath', 'sigh', 'cough', 'panting', 'shouting', 'crying'];

// Python str.isspace() for one code point.
export function pyIsSpace(cp: number): boolean {
  return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp === 0x85 || cp === 0xa0 || cp === 0x1680 ||
    (cp >= 0x2000 && cp <= 0x200a) || cp === 0x2028 || cp === 0x2029 || cp === 0x202f || cp === 0x205f || cp === 0x3000;
}

// Length in code points of a marker starting at cps[pos], or 0.
function markerAt(cps: string[], pos: number): number {
  if (cps[pos] !== '<') return 0;
  let i = pos + 1;
  if (cps[i] === '/') i++;
  let named = false;
  for (const tag of NONVERBAL_MARKUP_TAGS) {
    if (cps.slice(i, i + tag.length).join('') === tag) { i += tag.length; named = true; break; }
  }
  if (!named) return 0;
  while (i < cps.length && pyIsSpace(cps[i].codePointAt(0)!)) i++;
  if (cps[i] === '/') i++;
  if (cps[i] === '>') return i + 1 - pos;
  return 0;
}

// Code-point ranges [start, end) of the recognized markers, left to right, non-overlapping.
export function nonverbalMarkerSpans(text: string): [number, number][] {
  const cps = Array.from(text);
  const out: [number, number][] = [];
  for (let pos = 0; pos < cps.length;) {
    const n = markerAt(cps, pos);
    if (n) { out.push([pos, pos + n]); pos += n; } else pos++;
  }
  return out;
}

export function hasNonverbalMarkup(text: string): boolean {
  return nonverbalMarkerSpans(text).length > 0;
}

// Alternate text / marker pieces (even = text, odd = markers); their concatenation is `text`.
export function splitNonverbalMarkup(text: string): string[] {
  const cps = Array.from(text);
  const pieces: string[] = [];
  let prev = 0;
  for (const [a, b] of nonverbalMarkerSpans(text)) {
    pieces.push(cps.slice(prev, a).join(''), cps.slice(a, b).join(''));
    prev = b;
  }
  pieces.push(cps.slice(prev).join(''));
  return pieces;
}

// Remove the markers (keeping enclosed words) and collapse whitespace: " ".join(sub(" ", text).split()).
export function stripNonverbalMarkup(text: string): string {
  const spans = nonverbalMarkerSpans(text);
  if (spans.length === 0) return text;
  const cps = Array.from(text);
  let replaced = '';
  let prev = 0;
  for (const [a, b] of spans) {
    replaced += cps.slice(prev, a).join('') + ' ';
    prev = b;
  }
  replaced += cps.slice(prev).join('');
  const words: string[] = [];
  let cur = '';
  for (const c of replaced) {
    if (pyIsSpace(c.codePointAt(0)!)) { if (cur) words.push(cur); cur = ''; } else cur += c;
  }
  if (cur) words.push(cur);
  return words.join(' ');
}
