// The checkpoint's text markers (pronunciation groups, the start marker, pause markers) against the Python reference on
// the release text rows (fixtures/text-markers.json: zonos3.tts.superbpe.text_ids, span_follow.token_char_ranges and
// plan.validate_text_markers on gguf/backbone-ptq-edit.gguf's spec).
import { describe, expect, it } from 'vitest';
import { tokenCharRanges } from '../src/request/spanFollow';
import { checkTextMarkers, createTokenizer, parseTextMarkers } from '../src/tokenizer';
import ref from './fixtures/text-markers.json';

const tok = createTokenizer();
const markers = parseTextMarkers(JSON.stringify(ref.spec), ref.text_vocab, ref.text_extra_tokens);

describe('text markers (release text rows)', () => {
  it.each(ref.encode.map(c => [c.text, c] as const))('encodes %j as superbpe.text_ids', (text, c) => {
    expect(Array.from(tok.encode(text, markers))).toEqual(c.ids);
    const m = { phonemes: markers.phonemes, start: markers.start !== null, pauses: markers.pauses !== null };
    expect(tokenCharRanges(text, s => tok.charOffsets(s), m)).toEqual(c.ranges);
  });

  it.each(ref.errors.map(c => [c.text, c] as const))('validates %j as plan.validate_text_markers', (text, c) => {
    expect(checkTextMarkers(markers, text, 'edit' in c && c.edit === true)).toBe(c.error);
  });

  it('refuses a spec whose rows do not end the extra rows', () => {
    expect(() => parseTextMarkers(JSON.stringify(ref.spec), ref.text_vocab, ref.text_extra_tokens + 1))
      .toThrow('Phoneme / start / pause tokens do not match text_extra_tokens (they must be the last extra rows)');
  });
});
