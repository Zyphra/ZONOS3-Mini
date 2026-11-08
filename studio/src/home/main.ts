/**
 * The chooser at `/`: the Voice Studio (`/studio`) or the lite voice-cloning page (`/lite`). The page's query
 * (`?api=web`, `?server=…`) carries over to whichever one opens.
 */
import '@fontsource-variable/inter/wght.css';
import './home.css';
import { mountChrome } from '../lib/chrome';

mountChrome();
if (location.search) for (const a of document.querySelectorAll<HTMLAnchorElement>('a[data-keep-query]')) a.search = location.search;

/** Under the cards: whether a ZONOS server answers here, else what the apps will run on. */
async function probe() {
  const line = document.getElementById('server-line')!;
  const params = new URLSearchParams(location.search);
  const server = (params.get('server') || import.meta.env.VITE_ZONOS_SERVER_URL || '').replace(/\/+$/, '');
  const origin = server || location.origin;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 3000);
  try {
    const res = await fetch(`${server}/v2/capabilities`, { signal: ctrl.signal, headers: { accept: 'application/json' } });
    // A static host answers every path with the app's HTML page: only JSON is a ZONOS server.
    if (res.ok && /json/.test(res.headers.get('content-type') ?? '')) {
      line.textContent = `Server: ${origin}`;
      return;
    }
  } catch {
    /* no server, or not a ZONOS one */
  } finally {
    clearTimeout(t);
  }
  const gpu = 'gpu' in navigator;
  line.textContent = gpu ? 'No server — runs in this browser (one-time download ≈ 1.9 GB)' : 'No server and no WebGPU — placeholder audio';
}
void probe();
