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
  /**
   * Opens the card large; `returnTo` gets focus back on close. Clicking the header or anything in the card that is not
   * a control opens it; with `whole`, the entire card is one button and its body a read-only preview.
   */
  onExpand?: (card: HTMLElement, returnTo: HTMLElement) => void;
  whole?: boolean;
  children: ComponentChildren;
}

/** Pointer-downs here operate a control in place and never open the card. */
const CONTROL = 'button, input, select, textarea, a, summary, label, [role="slider"], .slider-track, .panel-lock';

/** A bench panel: family-coloured header, optional lock overlay explaining why it can't be used. */
export function Panel({ family, title, aside, locked, class: cls, onExpand, whole, children }: Props) {
  const card = useRef<HTMLElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  const isWhole = !!onExpand && !!whole;
  // The element that acts as the button: the card when whole, else its header.
  const button = onExpand && { tabIndex: 0, role: 'button' as const, 'aria-haspopup': 'dialog' as const, 'aria-label': `${title}: open larger`, title: 'Open larger' };
  const onKey = (e: KeyboardEvent) => {
    if (e.target !== e.currentTarget || (e.key !== 'Enter' && e.key !== ' ')) return;
    e.preventDefault();
    e.stopPropagation();
    onExpand!(card.current!, e.currentTarget as HTMLElement);
  };
  return (
    <section
      ref={card}
      class={`panel${locked ? ' is-locked' : ''}${onExpand ? ' is-expandable' : ''}${isWhole ? ' is-whole' : ''}${cls ? ` ${cls}` : ''}`}
      style={family ? { '--fam': hue(family) } : undefined}
      aria-label={isWhole ? undefined : title}
      {...(isWhole ? button : {})}
      onKeyDown={isWhole ? onKey : undefined}
      onPointerDown={
        onExpand
          ? (e) => {
              const t = e.target as Element;
              down.current = e.button === 0 && (isWhole || t.closest('.panel-head') || !t.closest(CONTROL)) ? { x: e.clientX, y: e.clientY } : null;
            }
          : undefined
      }
      onClick={
        onExpand
          ? (e) => {
              const d = down.current;
              down.current = null;
              if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 6) onExpand(card.current!, isWhole ? card.current! : head.current!);
            }
          : undefined
      }
    >
      <div ref={head} class="panel-head" {...(onExpand && !isWhole ? button : {})} onKeyDown={onExpand && !isWhole ? onKey : undefined}>
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
      <div class="panel-body" inert={locked || isWhole ? true : undefined}>
        {children}
      </div>
      {locked && <div class="panel-lock">{locked}</div>}
    </section>
  );
}
