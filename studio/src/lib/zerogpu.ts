/**
 * ZeroGPU on a Hugging Face Space: the huggingface.co page that frames a Space hands the frame the visitor's quota
 * token (`x-ip-token`) per request, through the handshake Gradio's own client does (`supports-zerogpu-headers`
 * once, then `zerogpu-headers` over a MessageChannel). Requests that run on the GPU carry it, so they count against
 * the visitor's ZeroGPU quota instead of the Space's anonymous one. Anywhere else (no huggingface.co parent frame)
 * nothing is added.
 */

const HANDSHAKE = 'supports-zerogpu-headers';
const REQUEST = 'zerogpu-headers';

/** The routes that take a GPU on the Space (`servers/space/app.py`). */
const GPU_ROUTE = /^\/v2\/generate$/;

const parentOrigin = typeof window !== 'undefined' && window.parent !== window && location.hostname.endsWith('.hf.space')
  ? 'https://huggingface.co'
  : null;

let supported = false;
if (parentOrigin) {
  window.addEventListener('message', (e) => {
    if (e.origin === parentOrigin && e.data === HANDSHAKE) supported = true;
  });
  window.parent.postMessage(HANDSHAKE, parentOrigin);
}

/** Renders here count against the visitor's ZeroGPU quota (the Space's parent page answered the handshake). */
export const onZeroGpu = () => supported;

/** The parent's headers for one GPU request; null when it has none (or doesn't answer within 5 s). */
function zeroGpuHeaders(): Promise<Record<string, string> | null> {
  if (!parentOrigin || !supported) return Promise.resolve(null);
  const { promise, resolve } = Promise.withResolvers<Record<string, string> | null>();
  const channel = new MessageChannel();
  const done = (headers: Record<string, string> | null) => {
    clearTimeout(timer);
    channel.port1.close();
    resolve(headers);
  };
  const timer = setTimeout(() => done(null), 5000);
  channel.port1.onmessage = ({ data }) => done(data && typeof data === 'object' ? (data as Record<string, string>) : null);
  window.parent.postMessage(REQUEST, parentOrigin, [channel.port2]);
  return promise;
}

/** `init` with the parent's ZeroGPU headers when `input` is a GPU route, else `init` as is. */
export async function withZeroGpu(input: RequestInfo | URL, init?: RequestInit): Promise<RequestInit | undefined> {
  if (!supported) return init;
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  if (!GPU_ROUTE.test(new URL(url, location.href).pathname)) return init;
  const extra = await zeroGpuHeaders();
  if (!extra) return init;
  const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  return { ...init, headers };
}
