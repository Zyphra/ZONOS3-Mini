// Cloudflare Worker: the studio (with web-engine/) from Workers Static Assets; /models/* redirects to the web/ folder of
// Zyphra/ZONOS3-Mini on the Hugging Face Hub, which serves the shards with CORS. There is no API server on this
// origin: /v1/* and /v2/* answer 503, so the studio's `auto` backend picks the in-browser server (a 404 there would
// read as a ZONOS server without /v2/controls, the fixture backend).
interface Env {
  ASSETS: Fetcher;
}

const PREFIX = '/models/';
const MODELS = 'https://huggingface.co/Zyphra/ZONOS3-Mini/resolve/main/web/';

export default {
  async fetch(req, env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname.startsWith('/v1/') || url.pathname.startsWith('/v2/')) {
      return Response.json({ detail: 'No ZONOS3 server on this origin: the studio runs ZONOS3 in the browser (?api=web).' },
        { status: 503, headers: { 'cache-control': 'no-store' } });
    }
    if (!url.pathname.startsWith(PREFIX)) return env.ASSETS.fetch(req);
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return new Response('method not allowed', { status: 405, headers: { allow: 'GET, HEAD' } });
    }
    const key = url.pathname.slice(PREFIX.length);
    if (!key || decodeURIComponent(key).split('/').includes('..')) return new Response('not found', { status: 404 });
    // The app caches shards itself in Cache Storage (CACHE_NAME in src/gpu/weights.ts); the redirect is not cached so
    // a new Hub commit is picked up by the next uncached load. no-referrer: the Hub's CDN answers 404 (without CORS
    // headers) to requests whose Referer is a workers.dev page, and the browser follows a redirect under the
    // redirect's Referrer-Policy.
    return new Response(null, {
      status: 302,
      headers: { location: MODELS + key, 'cache-control': 'no-store', 'referrer-policy': 'no-referrer' },
    });
  },
} satisfies ExportedHandler<Env>;
