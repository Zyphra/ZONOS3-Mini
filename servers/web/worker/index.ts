// Cloudflare Worker: the studio (with web-engine/) from Workers Static Assets; /models/* streamed from R2. There is no
// API server on this origin: /v1/* and /v2/* answer 503, so the studio's `auto` backend picks the in-browser server
// (a 404 there would read as a ZONOS server without /v2/controls, the fixture backend).
interface Env {
  ASSETS: Fetcher;
  MODELS: R2Bucket;
}

const PREFIX = '/models/';

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
    const key = decodeURIComponent(url.pathname.slice(PREFIX.length));
    if (!key || key.split('/').includes('..')) return new Response('not found', { status: 404 });
    const obj = await env.MODELS.get(key);
    if (!obj) return new Response('not found', { status: 404 });
    const headers = new Headers();
    obj.writeHttpMetadata(headers);
    headers.set('etag', obj.httpEtag);
    headers.set('content-length', String(obj.size));
    // The app caches shards itself in Cache Storage (zonos3-web-models-v1); keep the browser HTTP cache from storing a second copy.
    headers.set('cache-control', 'no-store');
    return new Response(req.method === 'HEAD' ? null : obj.body, { headers });
  },
} satisfies ExportedHandler<Env>;
