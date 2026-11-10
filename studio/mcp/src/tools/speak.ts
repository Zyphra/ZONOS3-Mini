/** zonos_speak: Direct lines with takes, rendered with the studio's own request composer. */
import { registerAppTool } from '@modelcontextprotocol/ext-apps/server';
import type { McpServer } from '@modelcontextprotocol/server';
import { randomInt } from 'node:crypto';
import { z } from 'zod';
import { langLabel } from '../../../src/lib/languages';
import { uid } from '../../../src/lib/math';
import { pronProblem } from '../../../src/lib/pron';
import type { Values } from '../../../src/state/engine';
import type { StoredTake } from '../../../src/state/projectSchema';
import { buildRequest, PREVIEW_SEED, remapSpans } from '../../../src/state/requests';
import type { Line } from '../../../src/state/script';
import { requestContext, sendable } from '../backend';
import { anchorSpans, resolveValues, resolveVoice } from '../design';
import { designSummary, generationParams, hasValues, takeText, wavSeconds } from '../params';
import type { TakeInfo } from '../payload';
import { openProject, projectLine, saveProject } from '../project';
import { state } from '../state';
import * as zonos from '../zonos';
import { guard, language, lineRef, PLAYER_META, project, seed, values, warningsText, widget } from './common';
import { loadCtx, requireSteering, withSession } from './context';

export function findLine(lines: Line[], ref: number | string): { line: Line; index: number } {
  const index = typeof ref === 'number' ? ref - 1 : lines.findIndex((l) => l.id === ref);
  const line = lines[index];
  if (!line) throw new Error(`No line ${ref} (the script has ${lines.length} line${lines.length === 1 ? '' : 's'}; see zonos_project action=open)`);
  return { line, index };
}

export function registerSpeakTool(server: McpServer) {
  registerAppTool(
    server,
    'zonos_speak',
    {
      title: 'Speak a line',
      description:
        'Render a line of speech into the project’s Direct script (new line, or `line` to add takes to an existing one). Each line has its own voice: `voice` sets this line’s voice (otherwise it keeps its voice, or follows the script voice / Cast bench). `delivery`: how the whole line is said; `spans`: delivery on specific words. Values are delivery controls from zonos_controls.',
      inputSchema: {
        text: z.string().optional().describe('The line text (new line, or replaces the text of `line`)'),
        line: lineRef.optional(),
        voice: z.string().optional().describe('Persona name/id or library voice id/label for this line (default: the line’s voice, else the script voice / Cast bench)'),
        delivery: values.optional(),
        spans: z.array(z.object({ text: z.string(), occurrence: z.number().int().min(1).optional(), values })).optional(),
        language: language.optional(),
        seed,
        takes: z.number().int().min(1).max(4).optional().describe('Takes to render (default 1)'),
        project,
      },
      _meta: PLAYER_META,
    },
    guard(async (a) => {
      if (!a.text?.trim() && a.line === undefined) throw new Error('Pass text or line');
      const ctx = await loadCtx();
      const warnings: string[] = [];
      const resolved = a.delivery ? resolveValues(ctx.spec, a.delivery, 'delivery') : null;
      if (resolved) warnings.push(...resolved.warnings);
      const delivery: Values | null = resolved?.values ?? null;
      const spanInput = (a.spans ?? []).map((s) => {
        const r = resolveValues(ctx.spec, s.values, 'delivery');
        warnings.push(...r.warnings);
        return { text: s.text, occurrence: s.occurrence, values: r.values };
      });
      requireSteering(ctx, hasValues(delivery ?? undefined) || spanInput.some((s) => hasValues(s.values)));
      const sess = await withSession(ctx, await openProject(a.project));
      warnings.unshift(...sess.warnings);
      const doc = sess.doc;
      const script = doc.direct.script;

      if (a.language && a.language !== doc.settings.language) {
        doc.settings.language = a.language;
        warnings.push(`Project language set to ${langLabel(a.language)}`);
      }

      let line: Line;
      let index: number;
      if (a.line !== undefined) {
        ({ line, index } = findLine(script.lines, a.line));
        if (a.text !== undefined && a.text !== line.text) {
          line.spans = remapSpans(line.spans, line.text, a.text);
          line.text = a.text;
        }
      } else {
        line = { id: uid(), text: a.text!.trim(), spans: [] };
        script.lines.push(line);
        index = script.lines.length - 1;
      }
      // Voice: per line (line.voice ?? script.voice ?? the Cast bench).
      const own = a.voice ? null : doc.cast.personas.find((p) => p.id === (line.voice ?? script.voice));
      const persona = own ?? resolveVoice(doc, ctx.speakers, a.voice);
      if (a.voice) line.voice = persona.id;
      if (delivery || spanInput.length) line.spans = anchorSpans(line.text, delivery, spanInput);
      const problem = pronProblem(line.text, ctx.caps.phoneme_tokens ?? null);
      if (problem) throw new Error(problem);

      const ctxReq = requestContext(ctx.caps, doc);
      const existing = doc.direct.takes[line.id] ?? [];
      const n = a.takes ?? 1;
      const made: { take: StoredTake; wav: Uint8Array }[] = [];
      let error: string | null = null;
      for (let k = 0; k < n; k++) {
        const s = k === 0 ? (a.seed ?? (existing.length ? randomInt(0, 1e6) : PREVIEW_SEED)) : randomInt(0, 1e6);
        const request = buildRequest(ctx.spec, ctxReq, persona.design, line.text, s, { extra: script.baseline, spans: line.spans, language: doc.settings.language });
        try {
          const g = await zonos.generate(sendable(request, ctx.steering));
          const audio = { ttfbMs: g.ttfbMs, totalMs: g.totalMs, ...(g.spanFollow ? { spanFollow: g.spanFollow } : {}), ...(g.decoder ? { decoder: g.decoder } : {}), media: '' };
          made.push({ take: { id: uid(), text: line.text, seed: s, request, audio, createdAt: Date.now(), starred: false }, wav: g.wav });
        } catch (e) {
          error = e instanceof Error ? e.message : String(e);
          break;
        }
      }
      if (!made.length) throw new Error(error ?? 'No take rendered');

      const snapshot = structuredClone({ line, persona, settings: doc.settings, index });
      const { result, project: ref } = await saveProject(sess, 'direct', (d, media) => {
        // On a conflict retry `d` is the fresh document: apply the same change to it.
        if (d !== doc) {
          if (!d.cast.personas.some((p) => p.id === snapshot.persona.id)) d.cast.personas.push(snapshot.persona);
          d.settings = snapshot.settings;
          const i = d.direct.script.lines.findIndex((l) => l.id === snapshot.line.id);
          if (i >= 0) d.direct.script.lines[i] = snapshot.line;
          else d.direct.script.lines.push(snapshot.line);
        }
        const list = [...(d.direct.takes[snapshot.line.id] ?? [])];
        for (const m of made) {
          m.take.audio.media = media.add(m.wav, 'audio/wav');
          list.push(m.take);
        }
        if (!list.some((t) => t.starred)) made[0].take.starred = true;
        d.direct.takes[snapshot.line.id] = list;
        return { index: d.direct.script.lines.findIndex((l) => l.id === snapshot.line.id) };
      });
      for (const m of made) state.takes.set(m.take.id, { wav: m.wav, media: m.take.audio.media });

      const lineNo = result.index + 1;
      const takes: TakeInfo[] = made.map((m, k) => ({
        id: m.take.id,
        label: `Take ${existing.length + k + 1}`,
        media: m.take.audio.media,
        seed: m.take.seed,
        durationSec: wavSeconds(m.wav),
        totalMs: m.take.audio.totalMs,
        ttfbMs: m.take.audio.ttfbMs,
        ...(m.take.audio.spanFollow ? { spanFollow: m.take.audio.spanFollow } : {}),
        starred: m.take.starred,
      }));
      const params = generationParams({ spec: ctx.spec, design: persona.design, speakers: ctx.speakers, persona, language: doc.settings.language, request: made[0].take.request, text: line.text, spans: line.spans });
      const body =
        [
          `Line ${lineNo} “${line.text}”`,
          ...takes.map((t) => `${takeText(t)}${t.starred ? ' ★' : ''}`),
          designSummary(ctx.spec, params, line.text.length),
          ...(error ? [`Stopped after ${made.length} of ${n} takes: ${error}`] : []),
          projectLine(ref),
        ].join('\n') + warningsText(warnings);
      return widget(body, { kind: 'speak', lineId: line.id, title: `Line ${lineNo}`, text: line.text, takes, params, project: ref, warnings: error ? [...warnings, error] : warnings });
    }),
  );
}
