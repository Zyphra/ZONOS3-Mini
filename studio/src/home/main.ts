/**
 * The chooser at `/`: the Voice Studio (`/studio`) or the lite voice-cloning page (`/lite`). The page's query
 * (`?api=web`, `?server=…`) carries over to whichever one opens.
 */
import '@fontsource-variable/inter/wght.css';
import './home.css';
import { mountChrome } from '../lib/chrome';
import { webEngineAvailable } from '../lib/web-engine';

mountChrome();
if (location.search) for (const a of document.querySelectorAll<HTMLAnchorElement>('a[data-keep-query]')) a.search = location.search;

/** Under the cards: whether a ZONOS server answers here, else what each app will run on. */
async function probe() {
  const line = document.getElementById('server-line')!;
  const lite = document.getElementById('choice-lite')!;
  line.textContent = 'Checking for a server…';
  const params = new URLSearchParams(location.search);
  const server = (params.get('server') || import.meta.env.VITE_ZONOS_SERVER_URL || '').replace(/\/+$/, '');
  const origin = server || location.origin;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 3000);
  try {
    const res = await fetch(`${server}/v2/capabilities`, { signal: ctrl.signal, headers: { accept: 'application/json' } });
    // A static host answers every path with the app's HTML page: only JSON is a ZONOS server.
    if (res.ok && /json/.test(res.headers.get('content-type') ?? '')) {
      // A server without `/v2/controls` is the studio's fixture backend: it runs with its controls off.
      const controls = await fetch(`${server}/v2/controls`, { signal: ctrl.signal, headers: { accept: 'application/json' } }).then((r) => r.ok, () => false);
      line.textContent = controls ? `Server: ${origin}` : `Server: ${origin} · controls not applied`;
      return;
    }
  } catch {
    /* no server, or not a ZONOS one */
  } finally {
    clearTimeout(t);
  }
  if (await webEngineAvailable()) {
    line.textContent = 'No server · Voice Studio and Voice Cloning run in this browser (≈1.9 GB download once)';
    return;
  }
  line.textContent = 'No server · Voice Studio opens with demo audio · Voice Cloning unavailable';
  lite.setAttribute('aria-disabled', 'true');
  lite.setAttribute('title', 'Needs a ZONOS server, or a browser with WebGPU and the in-browser engine');
}
void probe();
