/**
 * Synthetic speaker embeddings for the fixture backends: a seeded base orthogonal to every raw direction, plus
 * the voice's prior doses along the real directions. Projections of these vectors onto the raw axes therefore
 * read back the priors, which places the default voices plausibly on the Who map.
 */
import type { ControlsSpec, Speaker } from '../api';
import voices from './voices.json';
import { dot, gaussian, hash32, norm, rng } from '../lib/math';
import { slug } from '../lib/files';

const BASE_NORM = 3.3; // ≈ ref_base_norm of the speaker_traits_mect60k set
const priorsByLabel = new Map((voices.voices as { label: string; priors: Record<string, number | undefined> }[]).map((v) => [v.label, v.priors]));

let basisFor: ControlsSpec | null = null;
let basis: Float64Array[] = [];

function orthoBasis(spec: ControlsSpec): Float64Array[] {
  if (basisFor === spec) return basis;
  basis = [];
  for (const d of Object.values(spec.directions)) {
    if (!d.vector) continue;
    const v = Float64Array.from(d.vector);
    for (const b of basis) {
      const p = dot(v, b);
      for (let i = 0; i < v.length; i++) v[i] -= p * b[i];
    }
    const n = norm(v);
    if (n > 1e-6) basis.push(v.map((x) => x / n));
  }
  basisFor = spec;
  return basis;
}

export function syntheticEmbedding(spec: ControlsSpec, id: string, label: string): Float32Array {
  const r = rng(hash32(id));
  const dim = spec.embedding_dim ?? 0;
  const e = new Float64Array(dim);
  for (let i = 0; i < dim; i++) e[i] = gaussian(r);
  for (const b of orthoBasis(spec)) {
    const p = dot(e, b);
    for (let i = 0; i < dim; i++) e[i] -= p * b[i];
  }
  const n = norm(e);
  for (let i = 0; i < dim; i++) e[i] *= BASE_NORM / n;
  const priors: Record<string, number | undefined> = { ...(priorsByLabel.get(label) ?? {
    // An uploaded voice: somewhere plausible, deterministic per id.
    'identity.gender': (r() < 0.5 ? -1 : 1) * (0.6 + 0.5 * r()),
    'identity.age': gaussian(r) * 0.7,
    'identity.pitch': gaussian(r) * 0.4,
  }) };
  // Gender / age at the voice's measured Who-map spot (ui.who), as doses along the real axes.
  const who = spec.ui.who;
  const m = who?.landmarks[label];
  if (who && m) {
    const l = Math.log(Math.max(1e-6, m.female) / Math.max(1e-6, 1 - m.female));
    priors['identity.gender'] = Math.max(-who.gender.clip_logit, Math.min(who.gender.clip_logit, l)) / who.gender.logit_per_dose;
    priors['identity.age'] = (m.age - who.age.center_years) / who.age.years_per_dose;
  }
  for (const [dirId, dose] of Object.entries(priors)) {
    const v = spec.directions[dirId]?.vector;
    if (!v || !dose) continue;
    for (let i = 0; i < dim; i++) e[i] += dose * v[i];
  }
  return Float32Array.from(e);
}

/** The default voices as `GET /v2/speakers` lists them. */
export function fixtureSpeakers(spec: ControlsSpec): Speaker[] {
  return voices.voices.map((v) => ({
    id: `fixture_${slug(v.label)}`,
    label: v.label,
    source_type: 'audio',
    scope: 'default',
    is_default: true,
    original_name: `${v.label.replaceAll(' ', '')}.mp3`,
    created_at: 0,
    has_preview: true,
    speaker_modes: ['embedding', 'reference'],
    embedding_dim: spec.embedding_dim,
    transcript: null,
    disabled_controls: [],
  }));
}
