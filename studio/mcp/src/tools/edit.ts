/** Speech editing: zonos_edit_speech (takes of a changed text), zonos_keep_edit (keep one as a new Edit version). */
import { registerAppTool } from '@modelcontextprotocol/ext-apps/server';
import type { McpServer } from '@modelcontextprotocol/server';
import { z } from 'zod';
import type { Dose, EditRequest } from '../../../src/api';
import { EDIT_LANGUAGES, langLabel } from '../../../src/lib/languages';
import { uid } from '../../../src/lib/math';
import { parsePron } from '../../../src/lib/pron';
import type { Source } from '../../../src/state/edit';
import { controlDose } from '../../../src/state/engine';
import type { ProjectDoc, StoredVersion } from '../../../src/state/projectSchema';
import { remapSpans } from '../../../src/state/requests';
import { caps } from '../backend';
import { config } from '../config';
import { resolveValues } from '../design';
import { samplingOf, takeText, wavSeconds } from '../params';
import type { GenerationParams, TakeInfo, WidgetPayload } from '../payload';
import { b64, isRiff, openProject, projectLine, saveProject, type ProjectRef } from '../project';
import { state, type EditDraft } from '../state';
import * as zonos from '../zonos';
import { editLanguage, guard, lineRef, PLAYER_URI, project, seed, text, values, warningsText, widget } from './common';
import { loadCtx, withSession } from './context';
import { findLine } from './speak';
import { readSource } from './voice';

const WAV_ONLY = 'Speech editing needs a WAV file (convert with: ffmpeg -i in.mp3 out.wav)';
const clipKey = (s: Source, seconds: number) => `${s.kind}:${s.label}:${seconds.toFixed(2)}`;
const fromB64 = (s: string) => new Uint8Array(Buffer.from(s, 'base64'));

/** WAV bytes of a take or version: this process's cache, else project media. */
async function audioOf(takeId: string | null, media: string): Promise<Uint8Array> {
  const hit = takeId ? state.takes.get(takeId) : undefined;
  return hit?.wav ?? zonos.getMedia(media);
}

function currentVersion(doc: ProjectDoc): StoredVersion {
  const e = doc.edit.doc;
  const v = e?.versions[e.current];
  if (!e || !v) throw new Error('The project has no Edit clip yet: pass source.line, source.file or source.url');
  return v;
}

function versionTake(v: StoredVersion, durationSec: number): TakeInfo {
  return { id: `v:${v.id}`, label: `Original (${v.label})`, media: v.media, seed: null, durationSec: Math.round(durationSec * 100) / 100, original: true };
}

export function registerEditTools(server: McpServer) {
  registerAppTool(
    server,
    'zonos_edit_speech',
    {
      title: 'Edit recorded speech',
      description:
        'Change words of recorded speech in its own voice: a Direct take (source.line), a WAV file/URL, or the current Edit clip. Returns takes to review; keep one with zonos_keep_edit.',
      inputSchema: {
        source: z.object({
          line: lineRef.optional(),
          take: z.string().optional().describe('Take id of the line (default: the starred, else last take)'),
          file: z.string().optional().describe('Local WAV file'),
          url: z.string().optional().describe('WAV URL'),
          current: z.boolean().optional().describe('The project’s current Edit clip version'),
        }),
        target_text: z.string().min(1).describe('The full new text of the clip'),
        transcript: z.string().optional().describe('What the file/url says (else it is transcribed)'),
        delivery: values.optional().describe('Delivery controls for the regenerated words'),
        takes: z.number().int().min(1).max(4).optional().describe('Takes to generate (default 2)'),
        seed,
        language: editLanguage.optional(),
        project,
      },
      _meta: { ui: { resourceUri: PLAYER_URI } },
    },
    guard(async (a) => {
      const c = await caps();
      if (!c.speech_editing) throw new Error('This server/checkpoint cannot edit speech (see docs/guide.md#speech-editing)');
      const ctx = await loadCtx();
      const sess = await withSession(ctx, await openProject(a.project));
      const doc = sess.doc;
      const warnings = [...sess.warnings];
      const src = a.source;
      const picks = [src.line !== undefined, !!src.file, !!src.url, !!src.current].filter(Boolean).length;
      if (picks !== 1) throw new Error('Pass exactly one source: line, file, url or current');
      if (parsePron(a.target_text, c.phoneme_tokens ?? null).groups.length) throw new Error('Pronunciation groups in edits are not supported here; use the Studio Edit screen');
      const lang = a.language ?? (EDIT_LANGUAGES.includes(doc.settings.language) ? doc.settings.language : 'en_us');

      let version: StoredVersion;
      let wav: Uint8Array;
      let source: Source;
      let ref: ProjectRef | null = null;
      if (src.current) {
        version = currentVersion(doc);
        wav = await zonos.getMedia(version.media);
        source = doc.edit.doc!.source;
      } else {
        let knownText: string | null;
        if (src.line !== undefined) {
          const { line, index } = findLine(doc.direct.script.lines, src.line);
          const list = doc.direct.takes[line.id] ?? [];
          const take = src.take ? list.find((t) => t.id === src.take) : (list.find((t) => t.starred) ?? list[list.length - 1]);
          if (!take) throw new Error(src.take ? `Line ${index + 1} has no take ${src.take}` : `Line ${index + 1} has no takes yet: render it with zonos_speak`);
          wav = await audioOf(take.id, take.audio.media);
          knownText = take.text;
          source = { kind: 'take', label: `Line ${index + 1} · take ${list.indexOf(take) + 1}`, lineId: line.id, takeId: take.id };
        } else {
          const { bytes, name } = await readSource(src.file, src.url);
          if (!isRiff(bytes)) throw new Error(WAV_ONLY);
          wav = bytes;
          knownText = a.transcript?.trim() || null;
          source = { kind: 'upload', label: name.replace(/\.[^.]+$/, '') };
        }
        const dur = wavSeconds(wav);
        const max = Math.min(c.edit_max_seconds, c.edit_align_max_seconds);
        if (dur < 0.3) throw new Error(`The clip is ${dur.toFixed(2)} s: editing needs at least 0.3 s`);
        if (dur > max) throw new Error(`Editing opens clips up to ${Math.round(max)} s (this one is ${dur.toFixed(1)} s): trim it first`);
        const al = knownText ? await zonos.align(b64(wav), knownText, lang) : await zonos.transcribe(b64(wav), lang);
        const opened = { wav, source, dur, al, knownText };
        const saved = await saveProject(sess, 'edit', (d, media) => {
          const v: StoredVersion = { id: uid(), label: 'Original', duration: opened.al.duration || opened.dur, transcript: opened.al.text, words: opened.al.words, alignedWords: opened.al.words, automatic: !opened.knownText, media: media.add(opened.wav, 'audio/wav') };
          d.edit.doc = { name: opened.source.label, source: opened.source, language: lang, versions: [v], current: 0 };
          d.edit.clipLanguages[clipKey(opened.source, opened.dur)] = lang;
          return v;
        });
        version = saved.result;
        ref = saved.project;
        if (version.automatic) warnings.push(`Transcribed as: “${version.transcript}” (check it; pass transcript to align known text instead)`);
      }

      const req: EditRequest = {
        audio_base64: b64(wav),
        original_text: version.transcript,
        target_text: a.target_text,
        margin_seconds: 0.08,
        num_takes: a.takes ?? 2,
        verify: true,
        seed: a.seed ?? null,
        language: lang,
      };
      if (a.delivery) {
        const r = resolveValues(ctx.spec, a.delivery, 'delivery');
        warnings.push(...r.warnings);
        const out: Dose = {};
        for (const cs of ctx.spec.ui.controls) if (cs.spannable && r.values[cs.id]) controlDose(cs, r.values[cs.id], out);
        for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 1e4) / 1e4;
        if (Object.keys(out).length) req.controls = out;
        if (Object.keys(out).length && !ctx.steering) throw new Error('This server has no steering bundle: delivery cannot be applied to edits.');
      }
      const res = await zonos.edit(req);
      const canSendBack = source.kind === 'take';
      const takes: TakeInfo[] = [versionTake(version, version.duration)];
      res.takes.forEach((t, k) => {
        const id = uid();
        const draft: EditDraft = { projectId: sess.id, versionId: version.id, take: t, target_text: a.target_text, language: lang, source };
        state.editTakes.set(id, draft);
        state.takes.set(id, { wav: fromB64(t.audio_base64) });
        t.stopped.forEach((s, r) => s === false && warnings.push(`Take ${k + 1}: region ${r + 1} never stopped (hit the length cap)`));
        takes.push({ id, label: `Take ${k + 1}`, seed: t.seed, durationSec: Math.round(t.duration * 100) / 100, verify: t.verify ? { wer: t.verify.wer, cer: t.verify.cer, asr_text: t.verify.asr_text } : null, keepable: true, canSendBack });
      });
      const requestShown: Record<string, unknown> = { ...req, audio_base64: `<wav, ${wav.length} bytes>` };
      const params: GenerationParams = {
        voice: { speakerId: '', label: 'the recording', mode: 'reference' },
        language: lang,
        values: [],
        accents: [],
        spans: [],
        doses: req.controls ?? {},
        controlSpans: [],
        identityEstimate: null,
        durationFactor: 1,
        sampling: samplingOf(requestShown, 'EditRequest'),
        request: requestShown,
      };
      const project = ref ?? (sess.id && sess.revision !== null ? { id: sess.id, name: doc.name, revision: sess.revision, url: `${config.server}/?project=${encodeURIComponent(sess.id)}#edit` } : null);
      const payload: WidgetPayload = { kind: 'edit', title: `Edit · ${source.label}`, text: a.target_text, takes, params, project, warnings };
      const body =
        [
          `Edit of ${source.label} (${langLabel(lang)}): “${version.transcript}” → “${a.target_text}”`,
          ...takes.slice(1).map((t) => takeText(t)),
          `Keep one with zonos_keep_edit take_id=<id>${canSendBack ? ' (send_back=true replaces the Direct take)' : ''}. Takes expire when this server restarts.`,
          ...(ref ? [projectLine(ref)] : []),
        ].join('\n') + warningsText(warnings);
      return widget(body, payload);
    }),
  );

  server.registerTool(
    'zonos_keep_edit',
    {
      title: 'Keep an edit take',
      description: 'Keep a take from zonos_edit_speech as the next Edit version; send_back=true also replaces the Direct take it came from.',
      inputSchema: { take_id: z.string(), send_back: z.boolean().optional() },
    },
    guard(async (a) => {
      const draft = state.editTakes.get(a.take_id);
      if (!draft) throw new Error('Unknown or expired take; generate again');
      const sess = await openProject(draft.projectId ?? undefined);
      if (currentVersion(sess.doc).id !== draft.versionId) throw new Error('The Edit clip changed since these takes; generate again');
      if (a.send_back && draft.source.kind !== 'take') throw new Error('Only takes from Direct can be sent back');
      const t = draft.take;
      const words = t.words ?? (await zonos.align(t.audio_base64, draft.target_text, draft.language)).words;
      const wav = fromB64(t.audio_base64);
      const { result, project: ref } = await saveProject(sess, a.send_back ? 'direct' : 'edit', (d, media) => {
        const e = d.edit.doc;
        if (!e || e.versions[e.current]?.id !== draft.versionId) throw new Error('The Edit clip changed since these takes; generate again');
        const sha = media.add(wav, 'audio/wav');
        const nv: StoredVersion = { id: uid(), label: 'edit', duration: t.duration, transcript: draft.target_text, words, alignedWords: words, automatic: false, media: sha };
        e.versions = e.versions.slice(0, e.current + 1).concat(nv);
        e.current += 1;
        if (a.send_back) {
          const { lineId, takeId } = draft.source;
          const line = d.direct.script.lines.find((l) => l.id === lineId);
          const list = lineId ? d.direct.takes[lineId] : undefined;
          if (!line || !list?.some((x) => x.id === takeId)) throw new Error('That take no longer exists in Direct');
          d.direct.takes[lineId!] = list.map((x) => (x.id === takeId ? { ...x, audio: { media: sha, ttfbMs: 0, totalMs: 0 }, text: draft.target_text, edited: true } : x));
          if (line.text !== draft.target_text) {
            line.spans = remapSpans(line.spans, line.text, draft.target_text);
            line.text = draft.target_text;
          }
        }
        return { n: e.current + 1, sha };
      });
      state.takes.set(a.take_id, { wav, media: result.sha });
      const msg = `Kept take ${a.take_id} as Edit version v${result.n} (“${draft.target_text}”)${a.send_back ? '; it replaced the original take in Direct' : ''}.`;
      return text(`${msg}\n${projectLine(ref)}`);
    }),
  );
}
