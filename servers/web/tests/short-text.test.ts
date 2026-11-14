// Short-text prompts against the Python reference on the release text rows (fixtures/short-text.json:
// zonos3.tts.short_text.prompt on gguf/backbone-ptq-edit.gguf's spec): which texts qualify, the prompt text and the
// filler's first text-span token.
import { describe, expect, it } from 'vitest';
import { shortTextPrompt } from '../src/request/shortText';
import { createTokenizer, parseTextMarkers } from '../src/tokenizer';
import markersRef from './fixtures/text-markers.json';
import ref from './fixtures/short-text.json';

const tok = createTokenizer();
const markers = parseTextMarkers(JSON.stringify(markersRef.spec), markersRef.text_vocab, markersRef.text_extra_tokens);

describe('short-text prompts (release text rows)', () => {
  it.each(ref.cases.map(c => [c.language, c.text, c] as const))('%s %j as short_text.prompt', (language, text, c) => {
    const got = shortTextPrompt(text, language, markers, s => tok.charOffsets(s), s => tok.encode(s, markers).length - 2);
    expect(got).toEqual(c.prompt === null ? null : [c.prompt, c.filler_token]);
  });
});
