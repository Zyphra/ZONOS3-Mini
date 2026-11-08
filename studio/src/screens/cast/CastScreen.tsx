import type { ComponentChildren } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { CardModal } from '../../components/CardModal';
import { preview, togglePreview } from '../../audio/preview';
import { Icon } from '../../components/Icon';
import { Panel } from '../../components/Panel';
import { focusFirst } from '../../components/focusTrap';
import { openRail, Rail, RailBar, RailToggle } from '../../components/Rail';
import { phone } from '../../lib/media';
import { activeHistory, primaryAction, spec } from '../../state/app';
import { baseSpeaker, canClone, design, designs, identity, resetDesign, roll, setMode, setValue } from '../../state/design';
import { identityWord } from '../../state/engine';
import type { History } from '../../state/history';
import { cast } from '../../state/library';
import { AccentAtlas } from './AccentAtlas';
import { LibraryRail } from './LibraryRail';
import { Personas, PreviewStrip, SavePanel, SpeakPanel, Variations } from './Monitor';
import { MoodControls, RoomAxis, SliderGroup, type Binding } from './QuickControls';
import { TextureRadar } from './TextureRadar';
import { AccentTile, DeliveryTile, MoodTile, RoomTile, WhoTile } from './Tiles';
import { WhoMap } from './WhoMap';

export const designBinding = (): Binding => ({
  values: design.value.values,
  begin: () => designs.begin(),
  set: setValue,
  end: () => designs.end(),
});

function IdentityMeter() {
  const id = identity.value;
  const thr = spec.value.ui.identity_threshold ?? 0.8;
  if (design.value.mode === 'reference') {
    return (
      <div class="identity-meter" title="Keep as is sends the recording itself: the identity is the recording's">
        <span class="eyebrow">identity</span>
        <span class="identity-caption">Same person (the recording)</span>
      </div>
    );
  }
  const word = identityWord(id, thr);
  const isNew = word === 'New person';
  return (
    <div
      class={`identity-meter${isNew ? ' is-new' : ''}`}
      title={`Identity estimate ${id.toFixed(2)} · below ${thr.toFixed(2)} reads as a different person. Product of each control’s measured identity curve: an estimate, not a measurement.`}
    >
      <span class="eyebrow">identity</span>
      <div class="meter">
        <div class="meter-fill" style={{ width: `${id * 100}%` }} />
        <div class="meter-mark" style={{ left: `${thr * 100}%` }} />
      </div>
      <span class="identity-caption">
        <b>{word}</b>
      </span>
    </div>
  );
}

/** ← / → in the mode radiogroup move between the two radios (the disabled one is skipped). */
function modeKeys(e: KeyboardEvent) {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const items = [...(e.currentTarget as HTMLElement).querySelectorAll<HTMLButtonElement>('[role="radio"]:not([disabled])')];
  const i = items.indexOf(document.activeElement as HTMLButtonElement);
  if (i < 0 || items.length < 2) return;
  e.preventDefault();
  const next = items[(i + (e.key === 'ArrowRight' ? 1 : items.length - 1)) % items.length];
  next.focus();
  next.click();
}

/** Phone: the right rail holds Save; this button opens it on the name field. */
function SaveShortcut() {
  if (!phone.value) return null;
  return (
    <button
      class="icon-btn"
      aria-label="Save this voice"
      title="Save this voice"
      onClick={() => {
        openRail('cast-monitor');
        requestAnimationFrame(() => {
          const rail = document.getElementById('rail-cast-monitor');
          if (rail) focusFirst(rail, '.save-panel');
        });
      }}
    >
      <Icon name="save" size={16} />
    </button>
  );
}

function BenchHeader() {
  const d = design.value;
  const clone = d.mode === 'reference';
  return (
    <div class="bench-head" id="bench-head">
      <div class="bench-title">
        <span class="eyebrow">Shape the voice</span>
        <h2 title={baseSpeaker.value?.label}>{baseSpeaker.value?.label ?? 'Pick a voice'}</h2>
      </div>
      <div class="mode-wrap">
        <div class="mode" role="radiogroup" aria-label="Voice mode" onKeyDown={modeKeys}>
          <button role="radio" aria-checked={!clone} class={`mode-btn${!clone ? ' is-on' : ''}`} onClick={() => setMode('embedding')} title="Shape it: who, accent and texture change the person">
            Shape it
          </button>
          <button
            role="radio"
            aria-checked={clone}
            class={`mode-btn${clone ? ' is-on' : ''}`}
            disabled={!canClone.value}
            onClick={() => setMode('reference')}
            title={canClone.value ? 'Keep as is: the recording itself; only delivery and mood change' : 'This voice was saved without a recording, so there is nothing to keep as is'}
          >
            Keep as is
          </button>
        </div>
        {!canClone.value && <span class="hint">Keep as is needs a recording; this voice was saved without one.</span>}
      </div>
      <IdentityMeter />
      <div class="row gap-s">
        <button class="btn" onClick={roll} title="A random voice inside every control’s natural range">
          <Icon name="dice" size={14} /> Roll a voice
        </button>
        <button class="icon-btn" onClick={resetDesign} title={clone ? 'Reset delivery and mood (Who, Accent, Texture and Room are kept for Shape it)' : 'Reset: back to the plain base voice'} aria-label="Reset the controls">
          <Icon name="refresh" size={16} />
        </button>
        <SaveShortcut />
      </div>
    </div>
  );
}

type CardId = 'who' | 'accent' | 'texture' | 'room' | 'delivery' | 'mood';
const CARD_TITLE: Record<CardId, string> = { who: 'Who', accent: 'Accent', texture: 'Texture', room: 'Room', delivery: 'Delivery', mood: 'Mood' };
/** The cards Exact clone locks (their controls live in the speaker embedding). */
const EMBEDDING_CARDS: CardId[] = ['who', 'accent', 'texture', 'room'];

const CLONE_LOCK = (
  <>
    <b>Keep as is copies the recording.</b>
    <span>Delivery and mood still work; Who, Accent, Texture and Room don’t. Switch to Shape it to change them; your moves here are kept.</span>
    <button class="btn sm" onClick={() => setMode('embedding')}>
      Switch to Shape it
    </button>
  </>
);
/** A phone tile has no room for the explanation: tapping the lock scrolls to the mode switch below the tiles. */
const TILE_LOCK = (
  <button class="link" onClick={() => document.getElementById('bench-head')?.scrollIntoView({ block: 'center', behavior: 'smooth' })}>
    <b>Keep as is</b> · change
  </button>
);

/**
 * First run: the right rail (starting points, variations, save) starts folded; it unfolds by itself the moment the
 * project gets its first take or cast member, and after that the reader's fold choice stands.
 */
function useFirstRunRail() {
  const hasWork = !!preview.value.a || cast.value.length > 0;
  const was = useRef(hasWork);
  useEffect(() => {
    if (hasWork && !was.current) openRail('cast-monitor');
    was.current = hasWork;
  }, [hasWork]);
  return hasWork;
}

export function CastScreen() {
  const hasWork = useFirstRunRail();
  useEffect(() => {
    activeHistory.value = designs as History<unknown>;
    primaryAction.value = togglePreview;
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
    };
  }, []);
  const b = designBinding();
  const onPhone = phone.value;
  const locked = design.value.mode === 'reference' ? CLONE_LOCK : undefined;
  const [open, setOpen] = useState<{ id: CardId; card: HTMLElement; returnTo: HTMLElement } | null>(null);
  const expand = (id: CardId) => (card: HTMLElement, returnTo: HTMLElement) => setOpen({ id, card, returnTo });
  // A card that locks (Exact clone) closes its dialog.
  useEffect(() => {
    if (open && locked && EMBEDDING_CARDS.includes(open.id)) setOpen(null);
  }, [!!locked]);
  // Phone: every control is a tile (a read-only summary) that opens its full control in the dialog.
  const tile = (id: CardId, children: ComponentChildren, lockable = true) => {
    const lock = lockable && locked ? TILE_LOCK : undefined;
    return (
      <Panel family={id} title={CARD_TITLE[id]} class="cast-tile" locked={lock} onExpand={lock ? undefined : expand(id)} whole>
        {children}
      </Panel>
    );
  };
  return (
    <div class="cast">
      <Rail id="cast-library" side="left" title="Voices">
        <LibraryRail />
      </Rail>
      <section class="bench" aria-label="Voice bench">
        <RailBar>
          <RailToggle id="cast-library" title="Voices" icon="mic" />
          <RailToggle id="cast-monitor" title="More" icon="sliders" />
        </RailBar>
        <SpeakPanel />
        {onPhone ? (
          <>
            <BenchHeader />
            <div class="cast-tiles">
              {tile('who', <WhoTile />)}
              {tile('accent', <AccentTile />)}
              {tile('texture', <TextureRadar />)}
              {tile('room', <RoomTile />)}
              {tile('delivery', <DeliveryTile />, false)}
              {tile('mood', <MoodTile />, false)}
            </div>
          </>
        ) : (
          <>
            <BenchHeader />
            <div class="hero">
              <Panel family="who" title="Who" locked={locked} onExpand={locked ? undefined : expand('who')} aside={<span class="hint" title="Drag the dot to move gender / age; scroll while the map has focus (or with ⌘ / Ctrl) for pitch; click a voice to start from it">drag the dot · scroll for pitch</span>}>
                <WhoMap />
              </Panel>
              <Panel family="accent" title="Accent" locked={locked} onExpand={locked ? undefined : expand('accent')} aside={<span class="hint">drag into a region · deeper = stronger</span>}>
                <AccentAtlas />
              </Panel>
            </div>
            <div class="strip">
              <Panel family="texture" title="Texture" locked={locked} onExpand={locked ? undefined : expand('texture')} whole>
                <TextureRadar />
              </Panel>
              <Panel family="room" title="Room" locked={locked} onExpand={locked ? undefined : expand('room')}>
                <RoomAxis b={b} />
              </Panel>
              <Panel family="delivery" title="Delivery" onExpand={expand('delivery')}>
                <SliderGroup group="delivery" b={b} compact />
              </Panel>
              <Panel family="mood" title="Mood" onExpand={expand('mood')}>
                <MoodControls b={b} compact padOpens />
              </Panel>
            </div>
          </>
        )}
      </section>
      <Rail id="cast-monitor" side="right" title="More" defaultFolded={!hasWork}>
        <aside class="monitor" aria-label="Monitor">
          <div class="monitor-scroll">
            <Personas />
            <Variations />
            <SavePanel />
          </div>
        </aside>
      </Rail>
      {open && (
        <CardModal
          key={open.id}
          family={open.id}
          title={CARD_TITLE[open.id]}
          origin={open.card}
          returnFocus={open.returnTo}
          onClose={() => setOpen(null)}
          aside={<IdentityMeter />}
          strip={<PreviewStrip />}
        >
          {open.id === 'who' ? (
            <WhoMap large />
          ) : open.id === 'accent' ? (
            <AccentAtlas large />
          ) : open.id === 'texture' ? (
            <TextureRadar large />
          ) : open.id === 'room' ? (
            <RoomAxis b={b} large />
          ) : open.id === 'delivery' ? (
            <SliderGroup group="delivery" b={b} detailed />
          ) : (
            <MoodControls b={b} large />
          )}
        </CardModal>
      )}
    </div>
  );
}
