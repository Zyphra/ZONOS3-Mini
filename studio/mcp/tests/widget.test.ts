import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import type { WidgetPayload } from '../src/payload';
import { PLAYER_META, PLAYER_URI, text, widget } from '../src/tools/common';

// Supported metadata forms used by Codex's resource URI resolver.
const hostMetadata = z.object({
  'openai/outputTemplate': z.string().optional().nullable(),
  ui: z.object({ resourceUri: z.string().optional() }).optional(),
  'ui/resourceUri': z.string().optional(),
});

const payload: WidgetPayload = {
  kind: 'speak', title: 'Line 3', text: 'Existing take', lineId: 'line3',
  takes: [{ id: 'u0tm30ou', label: 'Take 1', media: 'saved-media', seed: 1, durationSec: 14 }],
  project: null, warnings: [],
  params: {
    voice: { speakerId: 'british', label: 'BritishFemale', mode: 'embedding' },
    language: 'en_gb', values: [], accents: [], spans: [], doses: {}, controlSpans: [],
    identityEstimate: null, durationFactor: 1, sampling: {}, request: {},
  },
};

describe('player host compatibility', () => {
  it('supplies both the portable URI and the compatibility alias', () => {
    expect(hostMetadata.parse(PLAYER_META).ui?.resourceUri).toBe(PLAYER_URI);
    expect(PLAYER_META['openai/outputTemplate']).toBe(PLAYER_URI);
  });

  it('resolves the player from result metadata without a tool descriptor', () => {
    const result = widget('Saved take', payload);
    const meta = hostMetadata.parse(result._meta);
    expect(meta.ui?.resourceUri ?? meta['ui/resourceUri'] ?? meta['openai/outputTemplate']).toBe(PLAYER_URI);
    expect(result.structuredContent).toEqual(payload);
    expect(result.content).toEqual([{ type: 'text', text: 'Saved take' }]);
  });

  it('does not attach player metadata to ordinary text results', () => {
    expect(text('Status')._meta).toBeUndefined();
  });
});
