/** The Zonos player: takes with waveform and playback, every generation parameter, Regenerate / Keep / Open in Studio. */
import { App, applyDocumentTheme, applyHostFonts, applyHostStyleVariables, type McpUiHostContext } from '@modelcontextprotocol/ext-apps';
import type { CallToolResult } from '@modelcontextprotocol/server';
import { render } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import type { GenerationParams, TakeInfo, WidgetPayload } from '../src/payload';

const app = new App({ name: 'Zonos player', version: '0.1.0' });

// ── Audio ─────────────────────────────────────────────────────────────────────────────────────────

interface Loaded {
  url: string;
  peaks: Float32Array;
  duration: number;
}

const loaded = new Map<string, Promise<Loaded>>();
let audioCtx: AudioContext | null = null;

function isAudio(v: unknown): v is { base64: string; mime: string } {
  return !!v && typeof v === 'object' && 'base64' in v && typeof v.base64 === 'string' && 'mime' in v && typeof v.mime === 'string';
}

/** A take's WAV through the app-only tool, as a data: URL (the default MCP Apps CSP allows data: media) and peaks. */
function load(t: TakeInfo): Promise<Loaded> {
  let p = loaded.get(t.id);
  if (!p) {
    p = (async () => {
      const r = await app.callServerTool({ name: 'zonos_take_audio', arguments: { take_id: t.id, ...(t.media ? { media: t.media } : {}) } });
      if (r.isError || !isAudio(r.structuredContent)) throw new Error(resultText(r) || 'No audio');
      const { base64, mime } = r.structuredContent;
      const bin = atob(base64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      audioCtx ??= new AudioContext();
      const buf = await audioCtx.decodeAudioData(bytes.buffer);
      const ch = buf.getChannelData(0);
      const n = 600;
      const peaks = new Float32Array(n);
      const step = Math.max(1, Math.floor(ch.length / n));
      for (let i = 0; i < n; i++) {
        let m = 0;
        for (let j = i * step; j < Math.min(ch.length, (i + 1) * step); j++) m = Math.max(m, Math.abs(ch[j]));
        peaks[i] = m;
      }
      return { url: `data:${mime};base64,${base64}`, peaks, duration: buf.duration };
    })();
    p.catch(() => loaded.delete(t.id));
    loaded.set(t.id, p);
  }
  return p;
}

const player = new Audio();
let playingId: string | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((f) => f());
player.addEventListener('ended', () => {
  playingId = null;
  notify();
});
player.addEventListener('pause', notify);
player.addEventListener('play', notify);

function usePlayer() {
  const [, set] = useState(0);
  useEffect(() => {
    const f = () => set((x) => x + 1);
    listeners.add(f);
    return () => void listeners.delete(f);
  }, []);
}

function resultText(r: CallToolResult): string {
  return r.content.map((c) => (c.type === 'text' ? c.text : '')).join('\n');
}

/** A widget tool's `structuredContent` (written by this server's tools). */
function isPayload(v: unknown): v is WidgetPayload {
  return !!v && typeof v === 'object' && 'kind' in v && 'takes' in v && Array.isArray(v.takes) && 'params' in v;
}

// ── Take row ──────────────────────────────────────────────────────────────────────────────────────

function Wave({ data, active }: { data: Loaded | null; active: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const c = ref.current;
      if (!c) return;
      const w = c.clientWidth;
      const h = c.clientHeight;
      const dpr = devicePixelRatio || 1;
      if (c.width !== w * dpr) c.width = w * dpr;
      if (c.height !== h * dpr) c.height = h * dpr;
      const g = c.getContext('2d')!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      // The canvas's CSS `color` is var(--fg) resolved; custom properties read raw would be unresolved var() text.
      const fg = getComputedStyle(c).color || '#888';
      const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#5b6cff';
      const at = active && data ? player.currentTime / (data.duration || 1) : -1;
      if (data) {
        const n = data.peaks.length;
        let max = 0.01;
        for (const p of data.peaks) max = Math.max(max, p);
        for (let x = 0; x < w; x++) {
          const p = data.peaks[Math.min(n - 1, Math.floor((x / w) * n))] / max;
          const y = Math.max(1, p * (h / 2 - 2));
          g.fillStyle = x / w <= at ? accent : fg;
          g.globalAlpha = x / w <= at ? 1 : 0.45;
          g.fillRect(x, h / 2 - y, 1, y * 2);
        }
        g.globalAlpha = 1;
        if (at >= 0) {
          g.fillStyle = accent;
          g.fillRect(Math.round(at * w), 0, 2, h);
        }
      }
      if (active) raf = requestAnimationFrame(draw);
    };
    draw();
    const ro = new ResizeObserver(draw);
    if (ref.current) ro.observe(ref.current);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [data, active]);
  const seek = (e: MouseEvent) => {
    if (!data || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const f = (e.clientX - r.left) / r.width;
    if (!active) return;
    player.currentTime = f * data.duration;
  };
  return <canvas class="wave" ref={ref} onClick={seek} aria-label="Waveform (click to seek)" />;
}

function TakeRow({ t, preload, onKeep, busy }: { t: TakeInfo; preload: boolean; onKeep: (t: TakeInfo, sendBack: boolean) => void; busy: boolean }) {
  usePlayer();
  const [data, setData] = useState<Loaded | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const get = async () => {
    setLoading(true);
    try {
      const d = await load(t);
      setData(d);
      return d;
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (preload) void get();
  }, [t.id]);
  const active = playingId === t.id;
  const playing = active && !player.paused;
  const toggle = async () => {
    if (playing) return player.pause();
    if (active) return void player.play();
    // A click during the preload waits for it (load() shares one fetch per take) instead of being dropped.
    const d = data ?? (await get());
    if (!d) return;
    player.src = d.url;
    playingId = t.id;
    await player.play().catch((e) => setErr(String(e)));
    notify();
  };
  return (
    <li class={`take${t.original ? ' original' : ''}`}>
      <div class="take-meta">
        <button class={playing ? '' : 'primary'} onClick={() => void toggle()} aria-busy={loading} aria-label={playing ? `Pause ${t.label}` : `Play ${t.label}`}>
          {playing ? '❚❚ Pause' : loading ? '…' : '▶ Play'}
        </button>
        <b>{t.label}</b>
        {t.starred && <span class="star" title="Starred take">★</span>}
        <span class="muted">{t.durationSec.toFixed(2)} s</span>
        {t.seed != null && <span class="muted">seed {t.seed}</span>}
        {t.verify && <span class="muted" title={`Heard: ${t.verify.asr_text}`}>WER {(t.verify.wer * 100).toFixed(1)}%</span>}
        {t.totalMs != null && <span class="muted">{(t.totalMs / 1000).toFixed(1)} s render</span>}
        {t.spanFollow && <span class="muted">spans follow {t.spanFollow}</span>}
        {t.keepable && (
          <span class="row">
            <button disabled={busy} onClick={() => onKeep(t, false)}>Keep</button>
            {t.canSendBack && (
              <button disabled={busy} onClick={() => onKeep(t, true)}>
                Keep &amp; send back
              </button>
            )}
          </span>
        )}
      </div>
      {err ? <div class="error">{err}</div> : <Wave data={data} active={active} />}
    </li>
  );
}

// ── Parameters ────────────────────────────────────────────────────────────────────────────────────

const signed = (v: number) => (v > 0 ? `+${v}` : `${v}`);

function valueChip(v: GenerationParams['values'][number]) {
  const pole = v.poles ? (v.value < 0 ? v.poles[0] : v.poles[1]) : null;
  return (
    <span class={`chip ${v.zone}${v.dropped ? ' dropped' : ''}`} title={`${v.group} · ${v.zone}${v.dropped ? ' · dropped in reference mode' : ''}`}>
      {v.label} {signed(v.value)}
      {pole ? ` (${pole})` : ''}
    </span>
  );
}

function LineSpans({ text, spans }: { text: string; spans: GenerationParams['spans'] }) {
  const parts = [];
  let at = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start > at) parts.push(text.slice(at, s.start));
    parts.push(
      <span class="span" title={s.values.map((v) => `${v.label} ${signed(v.value)}`).join('\n')}>
        {text.slice(s.start, s.end)}
      </span>,
    );
    at = Math.max(at, s.end);
  }
  if (at < text.length) parts.push(text.slice(at));
  return <div class="line-text">{parts}</div>;
}

function Grid({ entries }: { entries: [string, unknown][] }) {
  return (
    <div class="grid">
      {entries.map(([k, v]) => [<span class="muted">{k}</span>, <span>{v === null || v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>])}
    </div>
  );
}

function RequestJson({ request }: { request: Record<string, unknown> }) {
  const pre = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(request, null, 2);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      const sel = getSelection();
      if (pre.current && sel) {
        const r = document.createRange();
        r.selectNodeContents(pre.current);
        sel.removeAllRanges();
        sel.addRange(r);
      }
    }
  };
  return (
    <>
      <button onClick={() => void copy()}>{copied ? 'Copied' : 'Copy'}</button>
      <pre ref={pre}>{json}</pre>
    </>
  );
}

function Params({ p, text, kind }: { p: GenerationParams; text: string; kind: WidgetPayload['kind'] }) {
  return (
    <>
      <section class="section">
        <h2>Voice</h2>
        <div class="row">
          {p.voice.persona && <b>{p.voice.persona.name}</b>}
          <span>{p.voice.label}</span>
          {kind !== 'edit' && <span class="chip">{p.voice.mode}</span>}
          <span class="muted">{p.language}</span>
        </div>
      </section>
      {p.values.length > 0 && (
        <section class="section">
          <h2>Design</h2>
          <div class="chips">{p.values.map(valueChip)}</div>
        </section>
      )}
      {p.accents.length > 0 && (
        <section class="section">
          <h2>Accent</h2>
          <div class="chips">
            {p.accents.map((a) => (
              <span class="chip">
                {a.label} {a.applied}
                {a.requested !== undefined ? ` (asked ${a.requested})` : ''}
              </span>
            ))}
          </div>
        </section>
      )}
      {p.spans.length > 0 && (
        <section class="section">
          <h2>Delivery</h2>
          <LineSpans text={text} spans={p.spans} />
        </section>
      )}
      {p.identityEstimate !== null && (
        <section class="section">
          <h2>Estimates</h2>
          <Grid
            entries={[
              ['identity kept', `${Math.round(p.identityEstimate * 100)}%`],
              ['duration', `×${p.durationFactor.toFixed(2)}`],
            ]}
          />
        </section>
      )}
      <section class="section">
        <h2>Sampling</h2>
        <Grid entries={Object.entries(p.sampling)} />
      </section>
      <details class="section">
        <summary>Steering doses</summary>
        {Object.keys(p.doses).length ? <Grid entries={Object.entries(p.doses)} /> : <span class="muted">none</span>}
        {p.controlSpans.map((s) => (
          <div>
            <div class="muted">
              control_spans [{s.start}, {s.end})
            </div>
            <Grid entries={Object.entries(s.controls)} />
          </div>
        ))}
      </details>
      <details class="section">
        <summary>Request</summary>
        <RequestJson request={p.request} />
      </details>
    </>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────────────────────────

interface View {
  payload: WidgetPayload | null;
  error: string | null;
}

let toolName: string | null = null;
let toolArgs: Record<string, unknown> = {};

function Player() {
  const [view, setView] = useState<View>({ payload: null, error: null });
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [linkFallback, setLinkFallback] = useState(false);

  useEffect(() => {
    app.ontoolinput = (p) => (toolArgs = p.arguments ?? {});
    app.ontoolresult = (r) => show(r);
    app.onhostcontextchanged = (ctx) => applyContext(ctx);
    void app.connect().then(() => {
      const ctx = app.getHostContext();
      if (ctx) applyContext(ctx);
      toolName = ctx?.toolInfo?.tool.name ?? null;
    });
  }, []);

  function show(r: CallToolResult) {
    if (r.isError) return setView({ payload: null, error: resultText(r) });
    const sc = r.structuredContent;
    if (isPayload(sc)) setView({ payload: sc, error: null });
    else setView({ payload: null, error: resultText(r) || 'No result' });
  }

  const p = view.payload;

  const regenerate = async () => {
    if (!p) return;
    const name = toolName ?? (p.kind === 'voice' ? 'zonos_design_voice' : 'zonos_speak');
    const args: Record<string, unknown> = { ...toolArgs, seed: Math.floor(Math.random() * 1e6) };
    if (name === 'zonos_speak' && p.lineId) {
      delete args.text;
      delete args.delivery;
      delete args.spans;
      args.line = p.lineId;
      args.takes = 1;
    }
    setBusy(true);
    setNote(null);
    try {
      const r = await app.callServerTool({ name, arguments: args });
      toolArgs = args;
      show(r);
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const keep = async (t: TakeInfo, sendBack: boolean) => {
    setBusy(true);
    setNote(null);
    try {
      const r = await app.callServerTool({ name: 'zonos_keep_edit', arguments: { take_id: t.id, send_back: sendBack } });
      const text = resultText(r);
      setNote(text);
      if (!r.isError) await app.updateModelContext({ content: [{ type: 'text', text }] }).catch(() => undefined);
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const open = async () => {
    if (!p?.project) return;
    if (!app.getHostCapabilities()?.openLinks) return setLinkFallback(true);
    try {
      const r = await app.openLink({ url: p.project.url });
      if (r.isError) setLinkFallback(true);
    } catch {
      setLinkFallback(true);
    }
  };

  if (view.error) return <div class="error">{view.error}</div>;
  if (!p) return <div class="muted">Waiting for the result…</div>;
  return (
    <>
      <div class="head">
        <h1>{p.title}</h1>
        <div class="row">
          {p.kind !== 'edit' && (
            <button onClick={() => void regenerate()} disabled={busy}>
              {busy ? 'Rendering…' : '↻ Regenerate'}
            </button>
          )}
          {p.project && <button onClick={() => void open()}>Open in Studio</button>}
        </div>
      </div>
      <p class="said">“{p.text}”</p>
      {linkFallback && p.project && (
        <div class="note">
          Open this in a browser: <span class="url">{p.project.url}</span>
        </div>
      )}
      {p.warnings.length > 0 && (
        <ul class="warnings">
          {p.warnings.map((w) => (
            <li>{w}</li>
          ))}
        </ul>
      )}
      {note && <div class="note">{note}</div>}
      <ul class="takes">
        {p.takes.map((t, i) => (
          <TakeRow key={t.id} t={t} preload={i === (p.kind === 'edit' ? 1 : 0)} onKeep={(x, s) => void keep(x, s)} busy={busy} />
        ))}
      </ul>
      <Params p={p.params} text={p.text} kind={p.kind} />
      {!p.project && <div class="note muted">Not saved: the server keeps no projects.</div>}
    </>
  );
}

function applyContext(ctx: McpUiHostContext) {
  if (ctx.theme) applyDocumentTheme(ctx.theme);
  if (ctx.styles?.variables) applyHostStyleVariables(ctx.styles.variables);
  if (ctx.styles?.css?.fonts) applyHostFonts(ctx.styles.css.fonts);
}

render(<Player />, document.getElementById('root')!);
