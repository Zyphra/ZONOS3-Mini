/**
 * Process-global state. The stateless HTTP transport builds one McpServer per request, so everything that must
 * outlive a request lives here: one active project per process, shared by every client of it.
 */
import { randomUUID } from 'node:crypto';
import type { EditTake } from '../../src/api';
import type { Source } from '../../src/state/edit';
import { config } from './config';

/** A bounded map that forgets the least recently used entry. */
export class Lru<K, V> {
  private map = new Map<K, V>();
  constructor(private readonly cap: number) {}
  get(k: K): V | undefined {
    const v = this.map.get(k);
    if (v !== undefined) {
      this.map.delete(k);
      this.map.set(k, v);
    }
    return v;
  }
  set(k: K, v: V) {
    this.map.delete(k);
    this.map.set(k, v);
    while (this.map.size > this.cap) this.map.delete(this.map.keys().next().value as K);
  }
}

export interface CachedAudio {
  wav: Uint8Array;
  /** The project media hash of the same bytes, when saved. */
  media?: string;
}

/** Edit takes not kept yet (Studio does not save draft takes either). */
export interface EditDraft {
  projectId: string | null;
  versionId: string;
  take: EditTake;
  target_text: string;
  language: string;
  source: Source;
}

export const state = {
  sessionId: 'mcp-' + randomUUID(),
  activeProjectId: config.project as string | null,
  takes: new Lru<string, CachedAudio>(200),
  editTakes: new Lru<string, EditDraft>(50),
  /** Saved voice id → live id per project (voices registered again after the server lost them). */
  voiceMaps: new Map<string, Map<string, string>>(),
};
