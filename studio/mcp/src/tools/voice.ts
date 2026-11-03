/** Voice tools: design (Cast persona + preview), clone from audio, save a designed voice as an embedding voice. */
import { registerAppTool } from '@modelcontextprotocol/ext-apps/server';
import type { McpServer } from '@modelcontextprotocol/server';
import { readFile } from 'node:fs/promises';
import { basename, extname } from 'node:path';
import { z } from 'zod';
import type { SpeakerMode } from '../../../src/api';
import { encodeNpy, slug } from '../../../src/lib/files';
import { uid } from '../../../src/lib/math';
import { homePin, mixAt, pinForMix } from '../../../src/state/atlas';
import type { Design } from '../../../src/state/design';
import { presetToValues, steerEmbedding, toDoses, type AccentMix } from '../../../src/state/engine';
import { EMBEDDING_TYPE } from '../../../src/state/media';
import { buildRequest, PREVIEW_SEED } from '../../../src/state/requests';
import { requestContext, sendable } from '../backend';
import { designView, findSpeaker, modeFor, resolveAccents, resolveValues, upsertPersona } from '../design';
import { designSummary, generationParams, takeText, wavSeconds } from '../params';
import type { TakeInfo } from '../payload';
import { b64, isRiff, openProject, projectLine, saveProject } from '../project';
import { state } from '../state';
import * as zonos from '../zonos';
import { guard, PLAYER_URI, project, seed, text, values, warningsText, widget } from './common';
import { loadCtx, requireSteering, withSession } from './context';

const MIME: Record<string, string> = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.m4a': 'audio/mp4' };

/** Bytes of a local file or a URL, with a file name. */
export async function readSource(path: string | undefined, url: string | undefined): Promise<{ bytes: Uint8Array; name: string }> {
  if (!!path === !!url) throw new Error('Pass exactly one of path or url');
  if (path) return { bytes: new Uint8Array(await readFile(path)), name: basename(path) };
  const res = await fetch(url!);
  if (!res.ok) throw new Error(`Could not fetch ${url}: ${res.status} ${res.statusText}`);
  return { bytes: new Uint8Array(await res.arrayBuffer()), name: basename(new URL(url!).pathname) || 'audio' };
}

export function registerVoiceTools(server: McpServer) {
  registerAppTool(
    server,
    'zonos_design_voice',
    {
      title: 'Design a voice',
      description:
        'Create or refine a Cast persona: base voice, mode, preset, control values (see zonos_controls), accents. Renders a preview and saves the persona by name; call again with the same name to iterate.',
      inputSchema: {
        name: z.string().optional().describe('Persona name; an existing persona is refined. Default: "<base voice> variant".'),
        base_voice: z.string().optional().describe('Library voice id or label (required for a new persona)'),
        mode: z.enum(['embedding', 'reference']).optional().describe('embedding: designable (all controls); reference: exact clone (delivery/mood only)'),
        preset: z.string().optional().describe('A preset name from zonos_controls (sets values and accent, forces embedding mode)'),
        values: values.optional().describe('Control values by id; 0 removes one'),
        accents: values.optional().describe('Accent strengths by id; replaces the accent'),
        reset: z.boolean().optional().describe('Start the named persona over from the base voice'),
        text: z.string().optional().describe('Preview text (default: the persona signature or the bundle preview line)'),
        seed,
        project,
      },
      _meta: { ui: { resourceUri: PLAYER_URI } },
    },
    guard(async (a) => {
      const ctx = await loadCtx();
      requireSteering(ctx, !!a.preset || Object.values(a.values ?? {}).some(Boolean) || Object.values(a.accents ?? {}).some(Boolean));
      const sess = await withSession(ctx, await openProject(a.project));
      const doc = sess.doc;
      const warnings = [...sess.warnings];
      const existing = a.name ? doc.cast.personas.find((p) => p.name.toLowerCase() === a.name!.toLowerCase()) : undefined;
      const base = a.base_voice ? findSpeaker(ctx.speakers, a.base_voice) : undefined;
      if (a.base_voice && !base) throw new Error(`Unknown library voice '${a.base_voice}'. Voices: ${ctx.speakers.map((s) => s.label).join(', ')}`);
      let design: Design;
      if (existing && a.reset !== true) design = structuredClone(existing.design);
      else {
        const from = base ?? (existing ? ctx.speakers.find((s) => s.id === existing.design.baseId) : undefined);
        if (!from) throw new Error('base_voice is required for a new persona (see zonos_voices)');
        design = { baseId: from.id, mode: modeFor(from, doc.settings.voiceMode), values: {}, pin: homePin() };
      }
      if (base) design.baseId = base.id;
      const speaker = ctx.speakers.find((s) => s.id === design.baseId);
      if (!speaker) throw new Error(`The persona's base voice ${design.baseId} is gone from the server; pass base_voice`);
      let requested: AccentMix | undefined;
      if (a.preset) {
        const preset = ctx.spec.ui.presets[a.preset] ?? Object.entries(ctx.spec.ui.presets).find(([k]) => k.toLowerCase() === a.preset!.toLowerCase())?.[1];
        if (!preset) throw new Error(`Unknown preset '${a.preset}'. Presets: ${Object.keys(ctx.spec.ui.presets).join(', ')}`);
        const p = presetToValues(preset);
        design.values = p.values;
        design.pin = pinForMix(p.accents, ctx.spec.ui.accents, design.pin.tab);
        design.mode = 'embedding';
        requested = p.accents;
      }
      if (a.values) {
        const r = resolveValues(ctx.spec, a.values, 'design');
        warnings.push(...r.warnings);
        const next = { ...design.values };
        for (const id of Object.keys(a.values)) {
          if (r.values[id]) next[id] = r.values[id];
          else delete next[id];
        }
        design.values = next;
      }
      if (a.accents) {
        const r = resolveAccents(ctx.spec, a.accents);
        warnings.push(...r.warnings);
        design.pin = r.pin;
        requested = r.requested;
      }
      const mode: SpeakerMode = a.mode ?? (speaker.speaker_modes.includes(design.mode) ? design.mode : modeFor(speaker, doc.settings.voiceMode));
      if (!speaker.speaker_modes.includes(mode)) throw new Error(`${speaker.label} has no ${mode} mode`);
      design.mode = mode;

      const view = designView(ctx.spec, design);
      warnings.push(...view.warnings);
      const name = existing?.name ?? a.name ?? `${speaker.label} variant`;
      const previewText = a.text ?? (existing?.signature || ctx.spec.ui.preview_text);
      const request = buildRequest(ctx.spec, requestContext(ctx.caps, doc), design, previewText, a.seed ?? PREVIEW_SEED);
      const g = await zonos.generate(sendable(request, ctx.steering));
      const takeId = uid();
      state.takes.set(takeId, { wav: g.wav });

      const { result: persona, project: ref } = await saveProject(sess, 'cast', (d) => {
        const p = upsertPersona(d, name, structuredClone(design), a.text ?? existing?.signature ?? '');
        d.cast.design = structuredClone(design);
        if (a.text) d.cast.previewText = a.text;
        return p;
      });

      const params = generationParams({ spec: ctx.spec, design, speakers: ctx.speakers, persona, language: 'en_us', request, text: previewText, requestedAccents: requested, view });
      const take: TakeInfo = { id: takeId, label: 'Preview', seed: g.seed, durationSec: wavSeconds(g.wav), totalMs: g.totalMs, ttfbMs: g.ttfbMs };
      const body = [
        `${existing ? 'Updated' : 'Created'} persona “${persona.name}”: ${designSummary(ctx.spec, params, previewText.length)}`,
        `Identity kept ≈ ${Math.round(view.identity * 100)}% · duration ×${view.duration.toFixed(2)}`,
        takeText(take),
        projectLine(ref),
      ].join('\n') + warningsText(warnings);
      return widget(body, { kind: 'voice', title: `${persona.name} · preview`, text: previewText, takes: [take], params, project: ref, warnings });
    }),
  );

  server.registerTool(
    'zonos_clone_voice',
    {
      title: 'Clone a voice',
      description: 'Register a voice from an audio file or URL (wav, mp3, flac, ogg, m4a) and add it to the project as a persona. Use it as `voice`.',
      inputSchema: { path: z.string().optional().describe('Local audio file'), url: z.string().optional(), label: z.string().min(1), project },
    },
    guard(async (a) => {
      const { bytes, name } = await readSource(a.path, a.url);
      const mime = MIME[extname(name).toLowerCase()] ?? 'application/octet-stream';
      const speaker = await zonos.createSpeaker({ label: a.label, speaker_audio_base64: `data:${mime};base64,${b64(bytes)}`, speaker_audio_name: name });
      // The project keeps a clip to register the voice again after a server restart: WAV when available.
      let clip = { bytes, type: mime };
      if (!isRiff(bytes)) {
        const preview = await zonos.speakerPreview(speaker.id).catch(() => null);
        if (preview && isRiff(preview)) clip = { bytes: preview, type: 'audio/wav' };
      }
      const sess = await openProject(a.project);
      const { result: persona, project: ref } = await saveProject(sess, 'cast', (d, media) => {
        const sha = media.add(clip.bytes, clip.type);
        d.voices = [...d.voices.filter((v) => v.id !== speaker.id), { id: speaker.id, label: a.label, kind: 'clip', media: sha }];
        return upsertPersona(d, a.label, { baseId: speaker.id, mode: modeFor(speaker, d.settings.voiceMode), values: {}, pin: homePin() });
      });
      return text(`Cloned “${a.label}” as library voice ${speaker.id} (modes ${speaker.speaker_modes.join('/')}); persona “${persona.name}”.\n${projectLine(ref)}${warningsText(sess.warnings)}`);
    }),
  );

  server.registerTool(
    'zonos_save_voice',
    {
      title: 'Save a designed voice',
      description: 'Bake a persona’s voice design (embedding mode) into a new library voice: its steered embedding. Delivery and mood stay in the persona.',
      inputSchema: { persona: z.string().optional().describe('Persona name or id (default: the Cast bench persona)'), label: z.string().optional(), project },
    },
    guard(async (a) => {
      const ctx = await loadCtx();
      const sess = await withSession(ctx, await openProject(a.project, { create: false }));
      const doc = sess.doc;
      const bench = JSON.stringify(doc.cast.design);
      const persona = a.persona
        ? doc.cast.personas.find((p) => p.id === a.persona || p.name.toLowerCase() === a.persona!.toLowerCase())
        : doc.cast.personas.find((p) => JSON.stringify(p.design) === bench);
      if (!persona) throw new Error(`No persona ${a.persona ?? 'on the Cast bench'}. Personas: ${doc.cast.personas.map((p) => p.name).join(', ') || '(none)'}`);
      if (persona.design.mode !== 'embedding') throw new Error('Exact clones are already library voices');
      requireSteering(ctx, true);
      const d = persona.design;
      const e = Float32Array.from(await zonos.speakerEmbedding(d.baseId));
      const steered = steerEmbedding(ctx.spec, e, toDoses(ctx.spec, d.values, mixAt(d.pin, ctx.spec.ui.accents), 'embedding'));
      const label = a.label ?? persona.name;
      const speaker = await zonos.createSpeaker({
        label,
        speaker_embedding_base64: `data:application/octet-stream;base64,${b64(encodeNpy(steered))}`,
        speaker_embedding_name: `${slug(label)}.npy`,
      });
      const raw = new Uint8Array(steered.buffer, steered.byteOffset, steered.byteLength);
      const { project: ref } = await saveProject(sess, 'cast', (doc2, media) => {
        const sha = media.add(raw, EMBEDDING_TYPE);
        doc2.voices = [...doc2.voices, { id: speaker.id, label, kind: 'embedding', media: sha }];
      });
      return text(`Saved “${persona.name}” as library voice “${label}” (${speaker.id}, embedding). Use voice="${speaker.id}" or "${label}".\n${projectLine(ref)}${warningsText(sess.warnings)}`);
    }),
  );
}
