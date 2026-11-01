/**
 * Project media by content hash (SHA-256 hex): audio as WAV (16-bit unless exact bytes exist, e.g. what an Edit version sent to
 * the server), voice embeddings as raw float32, videos as the file. One stored copy however many projects, takes
 * or versions refer to it.
 */
import { encodeWav } from '../lib/files';
import { sha256, sha256Async } from '../lib/hash';

export interface MediaInfo {
  type: string;
  size: number;
}

export const EMBEDDING_TYPE = 'application/x-float32';

/** Hashes in the store with their sizes, kept in step by the project store. */
export const stored = new Map<string, number>();

/**
 * Hash of every media object seen (PCM arrays, clips, embeddings and files are never mutated in place): saving
 * again neither re-encodes nor re-hashes. Restored objects are primed with the hash they came from, since
 * re-encoding decoded 16-bit audio would not give the same bytes.
 */
const known = new WeakMap<object, string>();

export function prime(obj: object, hash: string) {
  known.set(obj, hash);
}

/** Collects the media a project snapshot refers to; `pending` holds those the store doesn't have yet. */
export class MediaSink {
  readonly info: Record<string, MediaInfo> = {};
  readonly pending = new Map<string, Blob>();

  /** A known object whose bytes are stored (or pending): no encoding. A known hash whose copy was deleted is redone. */
  private hit(key: object, type: string): string | null {
    const h = known.get(key);
    const size = h === undefined ? undefined : (stored.get(h) ?? this.pending.get(h)?.size);
    if (h === undefined || size === undefined) return null;
    this.info[h] = { type, size };
    return h;
  }

  private add(key: object, type: string, bytes: Uint8Array<ArrayBuffer>, blob?: Blob, h = sha256(bytes)): string {
    known.set(key, h);
    this.info[h] = { type, size: bytes.byteLength };
    if (!stored.has(h) && !this.pending.has(h)) this.pending.set(h, blob ?? new Blob([bytes], { type }));
    return h;
  }

  /** Audio as a WAV: `exact` gives the bytes when some exist, else 16-bit PCM. */
  audio(pcm: Float32Array, sr: number, exact?: () => ArrayBuffer): string {
    return this.hit(pcm, 'audio/wav') ?? this.add(pcm, 'audio/wav', new Uint8Array(exact ? exact() : encodeWav(pcm, sr)));
  }

  /** WAV bytes as they are (a voice's reference clip). */
  wav(buf: ArrayBuffer): string {
    return this.hit(buf, 'audio/wav') ?? this.add(buf, 'audio/wav', new Uint8Array(buf));
  }

  embedding(e: Float32Array): string {
    return this.hit(e, EMBEDDING_TYPE) ?? this.add(e, EMBEDDING_TYPE, new Uint8Array(e.slice().buffer));
  }

  /** A file (a video) is stored as itself. */
  async file(f: Blob): Promise<string> {
    const type = f.type || 'application/octet-stream';
    const hit = this.hit(f, type);
    if (hit) return hit;
    const bytes = new Uint8Array(await f.arrayBuffer());
    return this.add(f, type, bytes, f, await sha256Async(bytes));
  }
}
