import type { Download } from '../lib/web-engine';

/** The in-browser server's model download: label, bar and "x of y · rate · ETA" (`.dl` in styles/base.css). */
export function DownloadBar({ d }: { d: Download }) {
  const pct = Math.round(d.fraction * 100);
  return (
    <div class="dl" role="progressbar" aria-label={d.label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-valuetext={`${pct}% · ${d.detail}`}>
      <div class="dl-head">
        <span class="dl-label">{d.label}</span>
        <span class="dl-pct">{pct}%</span>
      </div>
      <div class="dl-track">
        <div class="dl-fill" style={{ width: `${d.fraction * 100}%` }} />
      </div>
      <span class="dl-detail">{d.detail}</span>
    </div>
  );
}
