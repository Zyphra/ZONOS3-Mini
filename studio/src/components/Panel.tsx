import type { ComponentChildren } from 'preact';
import { useRef } from 'preact/hooks';
import { hue } from '../state/app';
import { Icon } from './Icon';

interface Props {
  family?: string;
  title: string;
  aside?: ComponentChildren;
  locked?: ComponentChildren;
  class?: string;
  /** Clicking the header or the card's background (not a control) opens it large. */
  onExpand?: (card: HTMLElement, head: HTMLElement) => void;
  children: ComponentChildren;
}

/** Pointer-downs here operate a control in place and never open the card. */
const CONTROL = 'button, input, select, textarea, a, summary, label, [role="slider"], .slider-track, .radar-svg, .mood-pad, .panel-lock';

/** A bench panel: family-coloured header, optional lock overlay explaining why it can't be used. */
export function Panel({ family, title, aside, locked, class: cls, onExpand, children }: Props) {
  const card = useRef<HTMLElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  const open = () => onExpand?.(card.current!, head.current!);
  return (
    <section
      ref={card}
      class={`panel${locked ? ' is-locked' : ''}${onExpand ? ' is-expandable' : ''}${cls ? ` ${cls}` : ''}`}
      style={family ? { '--fam': hue(family) } : undefined}
      aria-label={title}
      onPointerDown={
        onExpand
          ? (e) => {
              const onHead = !!(e.target as Element).closest('.panel-head');
              down.current = e.button === 0 && (onHead || !(e.target as Element).closest(CONTROL)) ? { x: e.clientX, y: e.clientY } : null;
            }
          : undefined
      }
      onClick={
        onExpand
          ? (e) => {
              const d = down.current;
              down.current = null;
              if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 6) open();
            }
          : undefined
      }
    >
      <div
        ref={head}
        class="panel-head"
        tabIndex={onExpand ? 0 : undefined}
        role={onExpand ? 'button' : undefined}
        aria-haspopup={onExpand ? 'dialog' : undefined}
        aria-label={onExpand ? `${title}: open larger` : undefined}
        title={onExpand ? 'Open larger' : undefined}
        onKeyDown={
          onExpand
            ? (e) => {
                if (e.target !== e.currentTarget || (e.key !== 'Enter' && e.key !== ' ')) return;
                e.preventDefault();
                e.stopPropagation();
                open();
              }
            : undefined
        }
      >
        <h3>
          {family && <span class="fam-dot" />}
          {title}
        </h3>
        {aside && <div class="panel-aside">{aside}</div>}
        {onExpand && (
          <span class="panel-expand" aria-hidden="true">
            <Icon name="expand" size={14} />
          </span>
        )}
      </div>
      <div class="panel-body" inert={locked ? true : undefined}>
        {children}
      </div>
      {locked && <div class="panel-lock">{locked}</div>}
    </section>
  );
}
