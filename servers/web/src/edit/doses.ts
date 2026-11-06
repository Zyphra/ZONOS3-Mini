// Steering doses of a speech edit's regions (zonos3.tts.edit_runtime.region_doses).

export type Doses = Record<string, number>;

/** `controls` plus `region_controls[k]` summed per id for each region (zero doses dropped); null without any dose.
 *  Without explicit spans the edit has one dose set (applied to every diffed region). */
export function regionDoses(controls: Doses | null, regionControls: (Doses | null)[] | null,
                            spans: unknown[] | null): Doses[] | null {
  if (regionControls !== null) {
    if (spans === null) throw new Error('region_controls needs explicit spans (one entry per span).');
    if (regionControls.length !== spans.length)
      throw new Error(`region_controls has ${regionControls.length} entries for ${spans.length} spans; ` +
        'send one per span (null for none).');
  }
  const doses: Doses[] = [];
  for (let k = 0; k < (spans === null ? 1 : spans.length); k++) {
    const merged = new Map<string, number>(Object.entries(controls ?? {}));
    for (const [id, v] of Object.entries(regionControls?.[k] ?? {})) merged.set(id, (merged.get(id) ?? 0) + v);
    doses.push(Object.fromEntries([...merged].filter(([, v]) => v !== 0)));
  }
  return doses.some(d => Object.keys(d).length > 0) ? doses : null;
}
