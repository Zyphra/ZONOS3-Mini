import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GenerateRequest, Rendered, Speaker } from '../api';
import * as api from '../api';
import { projectSettings } from './app';
import { design, designs } from './design';
import { speakers } from './library';
import { addLine, failedLines, lineRequest, newScript, removeLine, renderLine, restoreLine, script, scripts, staleLines, takes, takeStaleReason, type Take } from './script';
import { useOfflineBackend } from '../test/backend';

// Hoisted above the imports: every render goes through this stub (a tiny silent clip, or a failure when asked).
vi.mock('../api', async (importOriginal) => {
  const mod = await importOriginal<typeof api>();
  let fail = false;
  const rendered = (req: GenerateRequest): Rendered => ({ pcm: new Float32Array(4410), sr: 44100, ttfbMs: 1, request: req, words: null }) as unknown as Rendered;
  return {
    ...mod,
    generate: vi.fn(async (req: GenerateRequest) => {
      if (fail) throw new Error('boom');
      return rendered(req);
    }),
    __setFail: (v: boolean) => (fail = v),
  };
});
const failNext = (v: boolean) => (api as unknown as { __setFail: (v: boolean) => void }).__setFail(v);

const voice: Speaker = { id: 'v1', label: 'Voice one', is_default: true, speaker_modes: ['embedding', 'reference'], has_preview: false, source_type: 'audio' } as Speaker;

beforeEach(() => {
  useOfflineBackend();
  speakers.value = [voice];
  designs.reset({ ...design.value, baseId: 'v1', mode: 'embedding' });
  scripts.reset(newScript());
  takes.value = {};
  failedLines.value = [];
  failNext(false);
});
afterEach(() => vi.restoreAllMocks());

describe('new scripts', () => {
  it('start with no lines', () => {
    expect(newScript().lines).toEqual([]);
  });
});

describe('takes', () => {
  it('the newest take is always the keeper', async () => {
    const id = addLine(null, 'Hello there');
    expect(await renderLine(id)).toBe(true);
    expect(await renderLine(id)).toBe(true);
    const list = takes.value[id];
    expect(list).toHaveLength(2);
    expect(list.map((t) => t.starred)).toEqual([false, true]);
  });

  it('a failed render stays in failedLines until the line renders again', async () => {
    // Its own text: the render cache is keyed by request, so another test's take must not serve this one.
    const id = addLine(null, 'This one fails first');
    failNext(true);
    expect(await renderLine(id)).toBe(false);
    expect(failedLines.value).toEqual([id]);
    failNext(false);
    expect(await renderLine(id)).toBe(true);
    expect(failedLines.value).toEqual([]);
  });

  it('a deleted line comes back with its takes', async () => {
    const a = addLine(null, 'First');
    const b = addLine(null, 'Second');
    await renderLine(b);
    const r = removeLine(b)!;
    expect(script.value.lines.map((l) => l.id)).toEqual([a]);
    restoreLine(r.line, r.index, r.takes);
    expect(script.value.lines.map((l) => l.id)).toEqual([a, b]);
    expect(takes.value[b]).toHaveLength(1);
  });
});

describe('gap gesture', () => {
  it('begin / set ×3 / end is one undo step', () => {
    const before = script.value.gap;
    scripts.begin();
    for (const v of [0.5, 0.7, 0.9]) scripts.set({ ...script.value, gap: v });
    scripts.end();
    expect(script.value.gap).toBe(0.9);
    expect(scripts.canUndo.value).toBe(true);
    scripts.undo();
    expect(script.value.gap).toBe(before);
    expect(scripts.canUndo.value).toBe(false);
  });
});

describe('staleness', () => {
  const takeFor = (lineId: string): Take => {
    const line = script.value.lines.find((l) => l.id === lineId)!;
    const request = lineRequest(line, 1);
    return { id: `t-${lineId}`, text: line.text, seed: 1, request, audio: { pcm: new Float32Array(10), sr: 44100, ttfbMs: 0, request, words: null } as unknown as Rendered, createdAt: 0, starred: true };
  };

  it('is computed once per take while nothing changed', () => {
    const ids = Array.from({ length: 5 }, (_, i) => addLine(null, `Line ${i}`));
    takes.value = Object.fromEntries(ids.map((id) => [id, [takeFor(id)]]));
    const spy = vi.spyOn(JSON, 'stringify');
    expect(staleLines.value).toEqual([]);
    const after = spy.mock.calls.length;
    // Reading again (another computed, another row) costs nothing more.
    for (const id of ids) takeStaleReason(script.value.lines.find((l) => l.id === id)!, takes.value[id][0]);
    expect(spy.mock.calls.length).toBe(after);
  });

  it('tells a reshaped voice from a baseline change', () => {
    const id = addLine(null, 'Hello');
    takes.value = { [id]: [takeFor(id)] };
    const line = () => script.value.lines.find((l) => l.id === id)!;
    expect(takeStaleReason(line(), takes.value[id][0])).toBeNull();
    // The script baseline moved: the stored take has no `pace` dose, the fresh one does.
    scripts.apply({ ...script.value, baseline: { pace: 1 } });
    expect(takeStaleReason(line(), takes.value[id][0])).toBe('baseline changed');
    scripts.apply({ ...script.value, baseline: {} });
    // The voice itself was reshaped in Cast (a non-baseline control).
    designs.apply({ ...design.value, values: { ...design.value.values, husky: 1 } });
    expect(takeStaleReason(line(), takes.value[id][0])).toBe('voice edited');
  });

  it('reads a decoder switch as a settings change, and an unserved decoder as the default', () => {
    const id = addLine(null, 'Which decoder?');
    const line = () => script.value.lines.find((l) => l.id === id)!;
    try {
      takes.value = { [id]: [takeFor(id)] };
      projectSettings.value = { ...projectSettings.value, decoder: 'full_mr' };
      expect(lineRequest(line(), 1).decoder).toBe('full_mr');
      expect(takeStaleReason(line(), takes.value[id][0])).toBe('decoder changed');
      takes.value = { [id]: [takeFor(id)] };
      expect(takeStaleReason(line(), takes.value[id][0])).toBeNull();
      // A project saved against a server with a decoder this one lacks renders on the default: no 400.
      projectSettings.value = { ...projectSettings.value, decoder: 'gone_head' };
      expect('decoder' in lineRequest(line(), 1)).toBe(false);
      expect(takeStaleReason(line(), takes.value[id][0])).toBe('decoder changed');
    } finally {
      projectSettings.value = { ...projectSettings.value, decoder: null };
    }
  });
});
