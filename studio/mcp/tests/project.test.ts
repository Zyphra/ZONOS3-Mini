import { describe, expect, it } from 'vitest';
import { remapIds, summaryOf } from '../../src/state/projectSchema';
import { MediaBag, newProjectDoc, sha256 } from '../src/project';

describe('project documents', () => {
  it('a take appended to a new project counts in the summary and registers its media', () => {
    const doc = newProjectDoc('p1', 'Test', 'en_us');
    const media = new MediaBag(doc);
    const wav = new Uint8Array([82, 73, 70, 70, 1, 2, 3, 4]);
    const sha = media.add(wav, 'audio/wav');
    doc.direct.script.lines.push({ id: 'l1', text: 'Hi.', spans: [] });
    doc.direct.takes.l1 = [{ id: 't1', text: 'Hi.', seed: 1, request: { text: 'Hi.' }, audio: { ttfbMs: 1, totalMs: 2, media: sha }, createdAt: 0, starred: true }];
    expect(summaryOf(doc).takes).toBe(1);
    expect(doc.media[sha]).toEqual({ type: 'audio/wav', size: wav.length });
    expect(sha).toBe(sha256(wav));
    expect(media.pending.get(sha)?.bytes).toBe(wav);
  });

  it('remapping saved → live → saved restores the document', () => {
    const doc = newProjectDoc('p1', 'Test', 'en_us');
    doc.cast.design.baseId = 'spk_saved1';
    doc.voices.push({ id: 'spk_saved1', label: 'Mine', kind: 'clip', media: 'a'.repeat(64) });
    doc.cast.origins.spk_saved1 = { x: 1, y: 2, measured: false };
    const json = JSON.stringify(doc);
    const map = new Map([['spk_saved1', 'spk_live9']]);
    const live = remapIds(json, map);
    expect(live).not.toContain('spk_saved1');
    expect(remapIds(live, new Map([['spk_live9', 'spk_saved1']]))).toBe(json);
  });
});
