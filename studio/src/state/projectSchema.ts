/**
 * The project document's shape (`schema` 1) and the pure helpers on it. Type imports only: the MCP server
 * (`studio/mcp`) imports this module in Node, where signals and the browser globals are not available.
 */
import type { Rendered, SpeakerMode } from '../api';
import type { LaciMode } from '../lib/laci';
import type { Design } from './design';
import type { Doc as EditDoc, Version } from './edit';
import type { Persona } from './library';
import type { MediaInfo } from './media';
import type { Script, Take } from './script';
import type { Spot } from './who';

export const SCHEMA = 1;

/**
 * Settings of the open project: the language (Direct renders, Edit clips, pronunciation lookups), the
 * default voice mode, the tag strength (tag guidance for lines with nonverbal tags; 1 = off), the audio
 * decoder (`/v2/capabilities.decoders` id; null = the server default; one the server lacks renders on its default),
 * and LACI (renders in a language without alignment heads on the server run without it).
 */
export interface ProjectSettings {
  language: string;
  voiceMode: SpeakerMode;
  tagStrength: number;
  decoder: string | null;
  laci: LaciMode;
}
export const DEFAULT_SETTINGS: ProjectSettings = { language: 'en_us', voiceMode: 'embedding', tagStrength: 1.5, decoder: null, laci: 'off' };

/** A server voice the project created and the media that recreates it (WAV clip, or float32 embedding). */
export interface VoiceRecord {
  id: string;
  label: string;
  kind: 'clip' | 'embedding';
  media: string;
}

export type StoredAudio = Omit<Rendered, 'pcm' | 'sr'> & { media: string };
export type StoredTake = Omit<Take, 'audio'> & { audio: StoredAudio };
export type StoredVersion = Omit<Version, 'pcm' | 'sr' | 'b64'> & { media: string };
export type StoredEditDoc = Omit<EditDoc, 'versions'> & { versions: StoredVersion[] };

/** What a project holds, for lists (the server's `GET /v2/projects` copies it from the document). */
export interface ProjectSummary {
  voices: number;
  cast: number;
  lines: number;
  takes: number;
  versions: number;
}

export function summaryOf(doc: ProjectDoc): ProjectSummary {
  return {
    voices: doc.voices.length,
    cast: doc.cast.personas.length,
    lines: doc.direct.script.lines.length,
    takes: Object.values(doc.direct.takes).reduce((n, t) => n + t.length, 0),
    versions: doc.edit.doc?.versions.length ?? 0,
  };
}

export interface ProjectDoc {
  schema: number;
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  settings: ProjectSettings;
  voices: VoiceRecord[];
  cast: { design: Design; personas: Persona[]; origins: Record<string, Spot>; previewText: string };
  /** Takes by line id (lines of the script only). */
  direct: { script: Script; takes: Record<string, StoredTake[]> };
  /**
   * The retired Scene act (older documents only; never written): its lines and their last renders, by line id.
   * Restore appends them to the script.
   */
  scene?: { scene: { lines: { id: string; characterId: string; text: string }[]; gap: number }; renders: Record<string, { key: string; audio: StoredAudio }> };
  edit: { doc: StoredEditDoc | null; clipLanguages: Record<string, string> };
  /** Every media hash the document refers to (SHA-256 hex; the server checks they exist before it stores the document). */
  media: Record<string, MediaInfo>;
  /** Written on save for project lists. */
  summary?: ProjectSummary;
}

export type ProjectHead = Pick<ProjectDoc, 'id' | 'name' | 'createdAt' | 'updatedAt'>;

/**
 * Replace server voice ids everywhere in a document's JSON: ids are random tokens (`spk_…`, offline `local_…`),
 * so a whole-token match only hits them — as values, map keys (voice origins) and inside derived strings (the
 * requests of takes and render keys), which keeps every take and render matching its line.
 */
export function remapIds(json: string, map: Map<string, string>): string {
  if (!map.size) return json;
  const esc = [...map.keys()].map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return json.replace(new RegExp(`(?<![\\w-])(${esc.join('|')})(?![\\w-])`, 'g'), (id) => map.get(id)!);
}
