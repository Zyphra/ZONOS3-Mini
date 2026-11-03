import { signal } from '@preact/signals';
import { useEffect, useRef } from 'preact/hooks';
import { exportWav } from '../../audio/mix';
import * as player from '../../audio/player';
import { nowPlaying, playhead } from '../../audio/player';
import { errorText, isAbort, render } from '../../audio/render';
import { Icon } from '../../components/Icon';
import { isPronError, PronMarks, PronounceButton } from '../../components/Pron';
import { Waveform } from '../../components/Waveform';
import { clamp, uid } from '../../lib/math';
import { activeHistory, act, backend, controlById, notify, primaryAction, requestContext, spec } from '../../state/app';
import { dubScene } from '../../state/dub';
import { accentsOf, design } from '../../state/design';
import type { History } from '../../state/history';
import { cast, savePersona, speakers, type Persona } from '../../state/library';
import { buildRequest, PREVIEW_SEED } from '../../state/requests';
import {
  addSceneLine,
  lineState,
  moveSceneLine,
  removeSceneLine,
  renderScene,
  renderSceneLine,
  scene,
  scenes,
  mixdown,
  setGap,
  timeline,
  updateSceneLine,
  sceneRequest,
  type SceneLine,
} from '../../state/scene';
import { extent, spotOf, who } from '../../state/who';
import { canEdit, openClip } from '../../state/edit';
import { LanguageSelect } from '../../components/LanguageSelect';
import { guidanceNote, TagButton, TagField, TagMarks, TagStrength } from '../../components/Tags';

const SCENE_ID = 'scene';
/** Characters tell apart by the data hues (theme-aware tokens). */
const CHAR_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c6)', 'var(--c3)', 'var(--c5)', 'var(--c4)', 'var(--o-hi)', 'var(--o)'];
export const charColor = (id: string) => CHAR_COLORS[Math.max(0, cast.value.findIndex((p) => p.id === id)) % CHAR_COLORS.length];

const signing = signal<Record<string, { busy: boolean; error: string | null }>>({});

async function playSignature(p: Persona) {
  const id = `sig:${p.id}`;
  if (player.isPlaying(id)) return player.stop();
  signing.value = { ...signing.value, [p.id]: { busy: true, error: null } };
  try {
    const r = await render(buildRequest(spec.value, requestContext.value, p.design, p.signature || spec.value.ui.preview_text, PREVIEW_SEED));
    signing.value = { ...signing.value, [p.id]: { busy: false, error: null } };
    player.play(id, r.pcm, r.sr);
  } catch (e) {
    signing.value = { ...signing.value, [p.id]: { busy: false, error: isAbort(e) ? null : errorText(e) } };
  }
}

function MiniWho({ p, color }: { p: Persona; color: string }) {
  const g = who.value;
  const base = spotOf(speakers.value.find((s) => s.id === p.design.baseId));
  const { x: [x0, x1], y: [y0, y1] } = extent.value;
  const fx = clamp((base.x + (p.design.values[g.gender.control] ?? 0) - x0) / (x1 - x0), 0, 1);
  const fy = clamp((base.y + (p.design.values[g.age.control] ?? 0) - y0) / (y1 - y0), 0, 1);
  return (
    <svg class="mini-who" width="48" height="48" viewBox="0 0 48 48" aria-label="Who-map position">
      <rect x="2" y="2" width="44" height="44" rx="8" />
      <line x1="24" x2="24" y1="4" y2="44" />
      <line y1="24" y2="24" x1="4" x2="44" />
      <circle cx={4 + fx * 40} cy={4 + (1 - fy) * 40} r="4.5" fill={color} />
    </svg>
  );
}

function CharacterCard({ p }: { p: Persona }) {
  const color = charColor(p.id);
  const base = speakers.value.find((s) => s.id === p.design.baseId);
  const mix = accentsOf(spec.value, p.design);
  const top = Object.entries(mix).sort((a, b) => b[1] - a[1])[0];
  const accent = top ? `${spec.value.ui.accents.find((a) => a.id === top[0])?.label ?? top[0]} ${top[1].toFixed(1)}` : 'no accent';
  const sig = signing.value[p.id];
  const playing = nowPlaying.value?.id === `sig:${p.id}`;
  return (
    <li class="char-card" style={{ '--char': color }}>
      <MiniWho p={p} color={color} />
      <div class="char-main">
        <b>{p.name}</b>
        <span class="sub">
          {base?.label ?? 'missing voice'} · {p.design.mode === 'reference' ? 'clone' : accent}
        </span>
        <span class="signature">“{p.signature || spec.value.ui.preview_text}”</span>
        {sig?.error && <span class="error-line small">{sig.error}</span>}
      </div>
      <div class="char-actions">
        <button class={`play-mini${playing ? ' is-playing' : ''}${sig?.busy ? ' is-busy' : ''}`} onClick={() => void playSignature(p)} aria-label={`Play ${p.name}'s signature line`}>
          <Icon name={playing ? 'stop' : 'play'} size={12} />
        </button>
        <button class="icon-btn sm" onClick={() => addSceneLine(p.id, '')} aria-label={`Add a line for ${p.name}`} title="Add a line for this character">
          <Icon name="plus" size={13} />
        </button>
      </div>
    </li>
  );
}

function DialogueLine({ l, n }: { l: SceneLine; n: number }) {
  const textEl = useRef<HTMLInputElement>(null);
  const st = lineState(l);
  const id = `line:${l.id}`;
  const playing = nowPlaying.value?.id === id;
  const p = cast.value.find((x) => x.id === l.characterId);
  const color = charColor(l.characterId);
  return (
    <li class="dline" style={{ '--char': color }}>
      <span class="line-n">{n}</span>
      <select class="input char-select" value={l.characterId} onChange={(e) => updateSceneLine(l.id, { characterId: (e.currentTarget as HTMLSelectElement).value })} aria-label={`Character for line ${n}`}>
        {!p && <option value={l.characterId}>(removed character)</option>}
        {cast.value.map((c) => (
          <option value={c.id}>{c.name}</option>
        ))}
      </select>
      <div class="dline-textcell">
        <div class="dline-textrow">
          <TagField text={l.text}>
            <input
              ref={textEl}
              class="input dline-text"
              value={l.text}
              placeholder="What do they say?"
              aria-label={`Line ${n} text`}
              data-tags
              onChange={(e) => updateSceneLine(l.id, { text: (e.currentTarget as HTMLInputElement).value })}
              onKeyDown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur()}
            />
          </TagField>
          <PronounceButton input={() => textEl.current} onDone={() => textEl.current?.blur()} />
          <TagButton input={() => textEl.current} />
        </div>
        <PronMarks text={l.text} replace={(s, e, t) => updateSceneLine(l.id, { text: l.text.slice(0, s) + t + l.text.slice(e) })} />
        <TagMarks text={l.text} replace={(next) => updateSceneLine(l.id, { text: next })} />
        {(l.at != null || (l.values && Object.keys(l.values).length > 0)) && (
          <div class="dline-tags">
            {l.at != null && (
              <button class="tag-btn" onClick={() => updateSceneLine(l.id, { at: undefined })} title="This line keeps its place in the scene (from a dub). Click to let it follow the previous line instead.">
                at {l.at.toFixed(2)} s ×
              </button>
            )}
            {l.values && Object.keys(l.values).length > 0 && (
              <button class="tag-btn" onClick={() => updateSceneLine(l.id, { values: undefined })} title="Direction on top of the character's voice (from a dub). Click to clear it.">
                {Object.entries(l.values)
                  .map(([k, v]) => `${controlById.value.get(k)?.label.toLowerCase() ?? k} ${v > 0 ? '+' : ''}${v}`)
                  .join(' · ')}{' '}
                ×
              </button>
            )}
          </div>
        )}
      </div>
      <div class="dline-audio">
        {st?.error ? (
          <span class="error-line small" title={st.error}>
            {isPronError(st.error) ? 'Not rendered: fix the pronunciation' : st.error}
          </span>
        ) : (
          <Waveform pcm={st?.audio?.pcm} color={color} playId={id} height={26} busy={st?.busy} />
        )}
        {st?.audio && guidanceNote(sceneRequest(l) ?? {}) && (
          <span class="take-follow is-tags" title={`${guidanceNote(sceneRequest(l) ?? {})}: the server takes one guidance per render`}>
            tags
          </span>
        )}
      </div>
      <div class="dline-actions">
        <button
          class={`play-mini${playing ? ' is-playing' : ''}${st?.busy ? ' is-busy' : ''}`}
          disabled={!st || st.busy}
          onClick={async () => {
            if (playing) return player.stop();
            const ready = st?.audio ?? (await renderSceneLine(l), lineState(l)?.audio);
            if (ready) player.play(id, ready.pcm, ready.sr);
          }}
          aria-label={`Play line ${n}`}
        >
          <Icon name={playing ? 'stop' : 'play'} size={12} />
        </button>
        <button class="icon-btn sm" onClick={() => moveSceneLine(l.id, -1)} aria-label={`Move line ${n} up`}>
          <Icon name="up" size={13} />
        </button>
        <button class="icon-btn sm" onClick={() => moveSceneLine(l.id, 1)} aria-label={`Move line ${n} down`}>
          <Icon name="down" size={13} />
        </button>
        {canEdit.value && (
          <button
            class="icon-btn sm"
            disabled={!st?.audio}
            onClick={() => st?.audio && void openClip({ kind: 'scene', label: `Scene line ${n}`, lineId: l.id }, st.audio.pcm, st.audio.sr, l.text)}
            aria-label={`Edit the words of line ${n}`}
            title={st?.audio ? 'Edit the words of this clip (Edit screen)' : 'Render the line first'}
          >
            <Icon name="edit" size={13} />
          </button>
        )}
        <button class="icon-btn sm" onClick={() => removeSceneLine(l.id)} aria-label={`Delete line ${n}`}>
          <Icon name="trash" size={13} />
        </button>
      </div>
    </li>
  );
}

async function playScene() {
  if (player.isPlaying(SCENE_ID)) return player.stop();
  if (!scene.value.lines.length) return;
  await renderScene();
  const clips = timeline.value.clips;
  if (clips.some((c) => !c.audio)) return notify('Some lines could not be rendered; fix them first (see the red notes).', 'error');
  const sr = clips[0].audio!.sr;
  player.play(SCENE_ID, mixdown(sr), sr);
}

async function exportScene() {
  await renderScene();
  const clips = timeline.value.clips.filter((c) => c.audio);
  if (!clips.length) return notify('Nothing to export yet.');
  const sr = clips[0].audio!.sr;
  exportWav(mixdown(sr), sr, 'scene');
}

function Timeline() {
  const { clips, total } = timeline.value;
  const lanes = [...new Set(clips.map((c) => c.line.characterId))];
  const playing = nowPlaying.value?.id === SCENE_ID;
  const at = playing ? playhead.value : null;
  if (!clips.length) return null;
  const pct = (t: number) => `${(t / (total || 1)) * 100}%`;
  return (
    <section class="timeline" aria-label="Timeline">
      <div class="timeline-head">
        <span class="eyebrow">Timeline</span>
        <span class="muted small">{total.toFixed(1)} s{clips.some((c) => !c.audio) && ' (estimated until rendered)'}</span>
        <label class="gap">
          gap
          <input type="range" min={0} max={1.5} step={0.05} value={scene.value.gap} onChange={(e) => setGap(Number((e.currentTarget as HTMLInputElement).value))} aria-label="Gap between lines (s)" />
          <b>{scene.value.gap.toFixed(2)} s</b>
        </label>
      </div>
      <div class="lanes">
        {lanes.map((cid) => {
          const p = cast.value.find((x) => x.id === cid);
          const color = charColor(cid);
          return (
            <div class="lane" style={{ '--char': color }}>
              <span class="lane-name">{p?.name ?? '—'}</span>
              <div class="lane-track">
                {clips
                  .filter((c) => c.line.characterId === cid)
                  .map((c) => (
                    <button
                      class={`clip${c.audio ? '' : ' is-pending'}`}
                      style={{ left: pct(c.start), width: pct(c.duration) }}
                      title={c.line.text}
                      onClick={() => c.audio && player.play(`line:${c.line.id}`, c.audio.pcm, c.audio.sr)}
                    >
                      {c.audio ? <Waveform pcm={c.audio.pcm} color={color} height={28} /> : <span>{c.line.text}</span>}
                    </button>
                  ))}
              </div>
            </div>
          );
        })}
        {at !== null && <div class="timeline-head-line" style={{ left: `calc(var(--lane-name) + (100% - var(--lane-name)) * ${at / (total || 1)})` }} />}
      </div>
    </section>
  );
}

export function SceneScreen() {
  useEffect(() => {
    activeHistory.value = scenes as History<unknown>;
    primaryAction.value = () => void playScene();
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
    };
  }, []);
  const people = cast.value;
  const lines = scene.value.lines;
  const busy = lines.some((l) => lineState(l)?.busy);
  const addBench = () => {
    const base = speakers.value.find((s) => s.id === design.value.baseId);
    const p: Persona = { id: uid(), name: `${base?.label ?? 'Voice'} (bench)`, design: structuredClone(design.value), signature: '', createdAt: Date.now() };
    savePersona(p);
    notify(`Added ${p.name} to the cast`, 'ok');
  };
  return (
    <div class="scene">
      <aside class="rail" aria-label="Characters">
        <div class="rail-section grow">
          <h4>Characters</h4>
          {people.length ? (
            <ul class="chars">{people.map((p) => <CharacterCard key={p.id} p={p} />)}</ul>
          ) : (
            <p class="muted small">Your cast is empty. Save voices in Cast; each becomes a character with its own voice and delivery.</p>
          )}
          <LanguageSelect />
          <TagStrength />
          <div class="col gap-s">
            <button class="btn" onClick={addBench} disabled={!design.value.baseId}>
              <Icon name="plus" size={14} /> Add the bench voice
            </button>
            <button class="btn ghost" onClick={() => (act.value = 'cast')}>
              Design a new voice in Cast
            </button>
          </div>
        </div>
      </aside>
      <section class="dialogue" aria-label="Dialogue">
        <header class="screen-head">
          <div>
            <span class="eyebrow">Scene</span>
            <h2>Dialogue</h2>
          </div>
          <div class="row gap-s" hidden={!people.length}>
            <button class={`btn${busy ? ' is-busy' : ''}`} disabled={!lines.length || busy} onClick={() => void renderScene()}>
              <Icon name="wand" size={14} /> {busy ? 'Rendering…' : 'Render all'}
            </button>
            <button class="btn primary" disabled={!lines.length} onClick={() => void playScene()}>
              <Icon name={nowPlaying.value?.id === SCENE_ID ? 'stop' : 'play'} size={14} /> {nowPlaying.value?.id === SCENE_ID ? 'Stop' : 'Play scene'}
            </button>
            <button class="btn" disabled={!lines.length} onClick={() => void exportScene()}>
              <Icon name="download" size={14} /> Export WAV
            </button>
            {backend.value?.dub && (
              <button class="btn" disabled={!lines.length || busy} onClick={() => void dubScene()} title="Dub this scene into another language: each line keeps its voice, direction and place (Dub screen)">
                <Icon name="globe" size={14} /> Dub this scene
              </button>
            )}
          </div>
        </header>
        {!people.length ? (
          <div class="empty">
            <Icon name="mic" size={26} />
            <h3>No characters yet</h3>
            <p>Add the voice on the Cast bench, or save a few voices to your cast first.</p>
            <button class="btn primary" onClick={addBench} disabled={!design.value.baseId}>
              Add the bench voice
            </button>
          </div>
        ) : (
          <>
            {lines.length ? (
              <ol class="dlines">
                {lines.map((l, i) => (
                  <DialogueLine key={l.id} l={l} n={i + 1} />
                ))}
              </ol>
            ) : (
              <p class="muted">Add a line for a character to start the scene.</p>
            )}
            <div class="add-lines">
              {people.map((p) => (
                <button class="btn ghost sm" style={{ '--char': charColor(p.id) }} onClick={() => addSceneLine(p.id, '', lines[lines.length - 1]?.id)}>
                  <span class="char-dot" /> {p.name} says…
                </button>
              ))}
            </div>
          </>
        )}
        <Timeline />
      </section>
    </div>
  );
}
