/** Read-mostly tools: status, controls vocabulary, voices, projects. */
import type { McpServer } from '@modelcontextprotocol/server';
import { z } from 'zod';
import { LANG_LABEL, langLabel } from '../../../src/lib/languages';
import { uid } from '../../../src/lib/math';
import { mixAt } from '../../../src/state/atlas';
import type { ProjectDoc, ProjectSummary } from '../../../src/state/projectSchema';
import { caps, spec } from '../backend';
import { config } from '../config';
import { newProjectDoc, NOT_SAVED, openProject, projectLine, saveProject, sessionName } from '../project';
import { state } from '../state';
import * as zonos from '../zonos';
import { guard, language, project, text, warningsText } from './common';
import { loadCtx, withSession } from './context';

const fmtTime = (ms: number | null | undefined) => (ms ? new Date(ms).toLocaleString('sv-SE').slice(0, 16) : '?');

/** `summary` of `GET /v2/projects` is the document's own, passed through untyped. */
function isSummary(s: unknown): s is ProjectSummary {
  return !!s && typeof s === 'object' && 'takes' in s && typeof s.takes === 'number';
}

/** A project's contents for the model: settings, cast, script, takes, Edit clip. */
function overview(doc: ProjectDoc, speakerLabel: (id: string) => string): string {
  const out: string[] = [];
  const st = doc.settings;
  out.push(`Project “${doc.name}” (${doc.id}) · ${langLabel(st.language)} · voice mode ${st.voiceMode} · tag strength ${st.tagStrength}`);
  out.push(doc.cast.personas.length ? 'Personas:' : 'Personas: none');
  for (const p of doc.cast.personas) {
    const vals = Object.entries(p.design.values).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(', ');
    out.push(`- ${p.name} (${p.id}): ${speakerLabel(p.design.baseId)}, ${p.design.mode}${vals ? ` · ${vals}` : ''}`);
  }
  const voice = doc.cast.personas.find((p) => p.id === doc.direct.script.voice);
  out.push(`Script voice: ${voice?.name ?? '(the Cast bench)'}`);
  doc.direct.script.lines.forEach((l, i) => {
    const takes = doc.direct.takes[l.id] ?? [];
    const ts = takes.map((t) => `${t.id} seed ${t.seed}${t.starred ? ' ★' : ''}${t.edited ? ' edited' : ''}`).join(', ');
    const lv = l.voice ? doc.cast.personas.find((p) => p.id === l.voice) : null;
    out.push(`${i + 1}. [${l.id}]${lv ? ` (${lv.name})` : ''} ${l.text}${l.spans.length ? ` (${l.spans.length} painted span${l.spans.length === 1 ? '' : 's'})` : ''}${ts ? ` — takes: ${ts}` : ''}`);
  });
  const e = doc.edit.doc;
  if (e) out.push(`Edit clip “${e.name}” (${e.source.kind}) · ${langLabel(e.language)} · versions: ${e.versions.map((v, i) => `v${i + 1} ${v.label}${i === e.current ? ' (current)' : ''}`).join(', ')}`);
  return out.join('\n');
}

export function registerInfoTools(server: McpServer) {
  server.registerTool(
    'zonos_status',
    { title: 'Zonos server status', description: 'What the Zonos server offers (steering, speech editing, projects, languages) and the active project.', inputSchema: {} },
    guard(async () => {
      const [c, s] = await Promise.all([caps(), spec()]);
      const lines = [
        `Server: ${config.server}`,
        `Steering: ${s.steering ? `live (${s.spec.ui.controls.length} controls, ${s.spec.ui.accents.length} accents)` : 'none (no steering bundle: plain voices only)'}`,
        `Speech editing: ${c.speech_editing ? `yes (clips up to ${Math.min(c.edit_max_seconds, c.edit_align_max_seconds)} s)` : 'no'}`,
        `Projects: ${c.projects.enabled ? 'enabled' : 'disabled (nothing is saved)'}`,
        `Text normalization languages: ${c.text_norm_languages.join(', ')}`,
        `Tag guidance: ${c.tag_cfg.available && c.tag_cfg.trained ? `yes (tags ${c.tag_cfg.tags.join(', ')}; scale ${c.tag_cfg.min_scale}..${c.tag_cfg.max_scale})` : 'no'}`,
        `Phoneme tokens (pronunciation groups): ${c.phoneme_tokens ? `yes (${c.phoneme_tokens.symbols.length} symbols)` : 'no'}`,
      ];
      let active = 'none';
      if (state.activeProjectId && c.projects.enabled) {
        try {
          const p = await zonos.getProject(state.activeProjectId);
          active = `“${p.name}” (${p.id}) rev ${p.revision} → ${config.server}/studio?project=${p.id}`;
        } catch {
          active = `${state.activeProjectId} (not saved yet)`;
        }
      }
      lines.push(`Active project: ${active}`);
      return text(lines.join('\n'));
    }),
  );

  server.registerTool(
    'zonos_controls',
    { title: 'Steering vocabulary', description: 'Every voice/delivery control with ranges and safe zones, accents, presets and moods. Call once before designing.', inputSchema: {} },
    guard(async () => {
      const { spec: s, steering } = await spec();
      const ui = s.ui;
      const out: string[] = [];
      if (!steering) out.push('NOTE: this server has no steering bundle; these controls cannot be applied.');
      out.push('Controls (id (group, kind) range, safe, [delivery] = also usable per line/span):');
      for (const c of ui.controls) {
        const kind = c.type === 'bipolar' ? `bipolar ${c.poles ? `${c.poles[0]}↔${c.poles[1]}` : ''}`.trim() : 'unipolar';
        out.push(`${c.id} (${c.group}, ${kind}) range ${c.range[0]}..${c.range[1]} safe ${c.safe[0]}..${c.safe[1]}${c.spannable ? ' [delivery]' : ''}${c.effect ? ` — ${c.effect}` : ''}`);
      }
      out.push('', 'Accents (one pin: the strongest wins):');
      for (const a of ui.accents) out.push(`${a.id} ${a.label} range ${a.range[0]}..${a.range[1]} safe ${a.safe[0]}..${a.safe[1]}${a.effect ? ` — ${a.effect}` : ''}`);
      out.push('', 'Presets:');
      for (const [name, p] of Object.entries(ui.presets)) out.push(`${name}: ${Object.entries(p).map(([k, v]) => `${k}=${v}`).join(' ')}`);
      if (ui.moods?.length) {
        out.push('', 'Moods (recommended dose):');
        for (const m of ui.moods) out.push(`${m.id} → ${m.control} at ${m.dose}`);
      }
      out.push('', `Preview text: ${ui.preview_text}`);
      return text(out.join('\n'), { controls: ui.controls, accents: ui.accents, presets: ui.presets, moods: ui.moods ?? [] });
    }),
  );

  server.registerTool(
    'zonos_voices',
    { title: 'Voices', description: 'Library voices (with measured gender/age hints) and the active project’s personas and created voices. Read-only.', inputSchema: { project } },
    guard(async ({ project: pid }) => {
      const ctx = await loadCtx();
      const who = ctx.spec.ui.who?.landmarks ?? {};
      const out = ['Library voices:'];
      for (const s of ctx.speakers) {
        const w = who[s.label];
        out.push(`- ${s.label} (${s.id}) · ${s.scope} · ${s.source_type} · modes ${s.speaker_modes.join('/')}${s.transcript ? ' · has transcript' : ''}${w ? ` · who: female ${w.female.toFixed(2)}, age ${Math.round(w.age)}` : ''}`);
      }
      if (pid || state.activeProjectId) {
        const sess = await withSession(ctx, await openProject(pid, { create: false }));
        const label = (id: string) => ctx.speakers.find((s) => s.id === id)?.label ?? id;
        out.push('', sess.id ? `Project “${sess.doc.name}” personas:` : 'Personas (this session, not saved):');
        if (!sess.doc.cast.personas.length) out.push('- none');
        for (const p of sess.doc.cast.personas) {
          const vals = Object.entries(p.design.values).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(', ');
          const acc = Object.entries(mixAt(p.design.pin, ctx.spec.ui.accents)).map(([k, v]) => `${k} ${v}`).join(', ');
          out.push(`- ${p.name} (${p.id}): ${label(p.design.baseId)}, ${p.design.mode}${vals ? ` · ${vals}` : ''}${acc ? ` · accent ${acc}` : ''}`);
        }
        if (sess.doc.voices.length) {
          out.push('Voices this project created:');
          for (const v of sess.doc.voices) out.push(`- ${v.label} (${sess.map.get(v.id) ?? v.id}) · ${v.kind}`);
        }
        out.push(warningsText(sess.warnings));
      }
      return text(out.join('\n').trim());
    }),
  );

  server.registerTool(
    'zonos_project',
    {
      title: 'Studio projects',
      description: 'List, open or create the Studio project the tools write into. open: by id or exact name; returns its personas, lines, takes and Edit clip.',
      inputSchema: {
        action: z.enum(['list', 'open', 'create']),
        id: z.string().optional(),
        name: z.string().optional(),
        language: language.optional(),
      },
    },
    guard(async ({ action, id, name, language: lang }) => {
      const c = await caps();
      if (!c.projects.enabled) return text(NOT_SAVED);
      if (action === 'list') {
        const list = await zonos.listProjects();
        if (!list.length) return text('No projects yet.');
        return text(
          list
            .map((p) => {
              const s = p.summary;
              const sum = isSummary(s) ? ` · ${s.cast} personas, ${s.lines} lines, ${s.takes} takes` : '';
              return `- ${p.name ?? '(unnamed)'} (${p.id}) · updated ${fmtTime(p.updated_at)}${sum}${p.id === state.activeProjectId ? ' · active' : ''}`;
            })
            .join('\n'),
        );
      }
      if (action === 'open') {
        let pid = id;
        if (!pid && name) pid = (await zonos.listProjects()).find((p) => p.name?.toLowerCase() === name.toLowerCase())?.id;
        if (!pid) throw new Error(`No project ${id ?? name ?? ''}: pass id or exact name (see action=list)`);
        const [sess, speakers] = await Promise.all([openProject(pid), zonos.speakers()]);
        const label = (sid: string) => speakers.find((s) => s.id === sid)?.label ?? sid;
        return text(`${overview(sess.doc, label)}\nRevision ${sess.revision} → ${config.server}/studio?project=${sess.id}${warningsText(sess.warnings)}`);
      }
      const nid = uid();
      const doc = newProjectDoc(nid, name ?? sessionName(), lang ?? 'en_us');
      const { project: ref } = await saveProject({ id: nid, doc, revision: null, map: new Map(), warnings: [] }, 'cast', () => undefined);
      return text(`Created project “${doc.name}” (${nid}), ${LANG_LABEL[doc.settings.language]}; it is now the active project.\n${projectLine(ref)}`);
    }),
  );
}
