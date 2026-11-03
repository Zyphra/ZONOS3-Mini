import { FILLED_ICONS, ICONS } from '../lib/marks';

export function Icon({ name, size = 16, filled, title }: { name: string; size?: number; filled?: boolean; title?: string }) {
  const fill = filled ?? !!FILLED_ICONS[name];
  return (
    <svg class="icon" width={size} height={size} viewBox="0 0 24 24" aria-hidden={title ? undefined : 'true'} role={title ? 'img' : undefined}>
      {title && <title>{title}</title>}
      <path d={ICONS[name] ?? ''} fill={fill ? 'currentColor' : 'none'} stroke={fill ? 'none' : 'currentColor'} stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  );
}
