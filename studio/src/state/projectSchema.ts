/**
 * The project document's shape (`schema` 1) and the pure helpers on it. Type imports only: the MCP server
 * (`studio/mcp`) imports this module in Node, where signals and the browser globals are not available.
 */
import type { Rendered, SpeakerMode } from '../api';
import type { Design } from './design';
import type { DubDoc, DubSource, DubTrack } from './dub';
import type { Doc as EditDoc, Version } from './edit';
import type { Persona } from './library';
import type { MediaInfo } from './media';
import type { Scene } from './scene';
import type { Script, Take } from './script';
import type { Spot } from './who';

export const SCHEMA = 1;

/**
 * Settings of the open project: the language (Direct / Scene renders, Edit clips, pronunciation lookups), the
 * default voice mode, and the tag strength (tag guidance for lines with nonverbal tags; 1 = off).
 */
export interface ProjectSettings {
  language: string;
  voiceMode: SpeakerMode;
  tagStrength: number;
}
export const DEFAULT_SETTINGS: ProjectSettings = { language: 'en_us', voiceMode: 'embedding', tagStrength: 1.5 };

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
export type StoredSource = Omit<DubSource, 'pcm' | 'sr' | 'video'> & { media: string; video: { media: string; name: string; type: string } | null };
/** `dubId` is always '': the server's dub (its takes) is gone with its session. */
export type StoredTrack = Omit<DubTrack, 'pcm' | 'sr'> & { media: string };
export type StoredDub = Omit<DubDoc, 'source' | 'refId' | 'track' | 'tracks'> & { source: StoredSource | null; trackId: string | null; tracks: StoredTrack[] };

/** What a project holds, for lists (the server's `GET /v2/projects` copies it from the document). */
export interface ProjectSummary {
  voices: number;
  cast: number;
  lines: number;
  takes: number;
  sceneLines: number;
  versions: number;
  dubTracks: number;
  video: boolean;
}

export function summaryOf(doc: ProjectDoc): ProjectSummary {
  return {
    voices: doc.voices.length,
    cast: doc.cast.personas.length,
    lines: doc.direct.script.lines.length,
    takes: Object.values(doc.direct.takes).reduce((n, t) => n + t.length, 0),
    sceneLines: doc.scene.scene.lines.length,
    versions: doc.edit.doc?.versions.length ?? 0,
    dubTracks: doc.dub.tracks.length,
    video: !!doc.dub.source?.video,
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
  /** The last render of each scene line, by line id. */
  scene: { scene: Scene; renders: Record<string, { key: string; audio: StoredAudio }> };
  edit: { doc: StoredEditDoc | null; clipLanguages: Record<string, string> };
  dub: StoredDub;
  /** Every media hash the document refers to (SHA-256 hex; the server checks they exist before it stores the document). */
  media: Record<string, MediaInfo>;
  /** Written on save for project lists. */
  summary?: ProjectSummary;
}

export type ProjectHead = Pick<ProjectDoc, 'id' | 'name' | 'createdAt' | 'updatedAt'>;

/**
 * Replace server voice ids everywhere in a document's JSON: ids are random tokens (`spk_…`, offline `local_…`),
 * so a whole-token match only hits them — as values, map keys (voice origins) and inside derived strings (the
 * requests of takes and render keys, dub voice keys), which keeps every take and render matching its line.
 */
export function remapIds(json: string, map: Map<string, string>): string {
  if (!map.size) return json;
  const esc = [...map.keys()].map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return json.replace(new RegExp(`(?<![\\w-])(${esc.join('|')})(?![\\w-])`, 'g'), (id) => map.get(id)!);
}
