import { useEffect, useState } from 'preact/hooks';
import { CardModal } from '../../components/CardModal';
import { togglePreview } from '../../audio/preview';
import { Icon } from '../../components/Icon';
import { Panel } from '../../components/Panel';
import { activeHistory, primaryAction, spec } from '../../state/app';
import { baseSpeaker, canClone, design, designs, identity, resetDesign, rollDesign, setMode, setValue } from '../../state/design';
import type { History } from '../../state/history';
import { AccentAtlas } from './AccentAtlas';
import { LibraryRail } from './LibraryRail';
import { Personas, PreviewSettings, PreviewStrip, SavePanel, SpeakPanel, Variations } from './Monitor';
import { MoodControls, RoomAxis, SliderGroup, type Binding } from './QuickControls';
import { TextureRadar } from './TextureRadar';
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
  return (
    <div class={`identity-meter${id < thr ? ' is-new' : ''}`} title="Estimated identity kept vs the base voice: product of each control’s measured identity curve. An estimate, not a measurement.">
      <span class="eyebrow">identity · est.</span>
      <div class="meter">
        <div class="meter-fill" style={{ width: `${id * 100}%` }} />
        <div class="meter-mark" style={{ left: `${thr * 100}%` }} />
      </div>
      <b>{id.toFixed(2)}</b>
    </div>
  );
}

function BenchHeader() {
  const d = design.value;
  const clone = d.mode === 'reference';
  return (
    <div class="bench-head">
      <div class="bench-title">
        <span class="eyebrow">Shape the voice</span>
        <h2 title={baseSpeaker.value?.label}>{baseSpeaker.value?.label ?? 'Pick a voice'}</h2>
      </div>
      <div class="mode" role="radiogroup" aria-label="Voice mode">
        <button role="radio" aria-checked={!clone} class={`mode-btn${!clone ? ' is-on' : ''}`} onClick={() => setMode('embedding')} title="Steer identity, accent, texture and room through the speaker embedding">
          Designable
        </button>
        <button
          role="radio"
          aria-checked={clone}
          class={`mode-btn${clone ? ' is-on' : ''}`}
          disabled={!canClone.value}
          onClick={() => setMode('reference')}
          title={canClone.value ? 'Clone the reference recording exactly; only delivery and mood can be steered' : 'This voice was saved from an embedding and has no recording to clone'}
        >
          Exact clone
        </button>
      </div>
      <IdentityMeter />
      <div class="row gap-s">
        <button class="btn" onClick={() => designs.apply(rollDesign())} title="A random voice inside every control’s tested-safe range">
          <Icon name="dice" size={14} /> Roll a voice
        </button>
        <button class="icon-btn" onClick={resetDesign} title="Reset: back to the plain base voice" aria-label="Reset to the plain base voice">
          <Icon name="refresh" size={16} />
        </button>
      </div>
    </div>
  );
}

type CardId = 'texture' | 'room' | 'delivery' | 'mood';
const CARD_TITLE: Record<CardId, string> = { texture: 'Texture', room: 'Room', delivery: 'Delivery', mood: 'Mood' };

const CLONE_LOCK = (
  <>
    <b>Exact clone keeps the recording’s identity.</b>
    <span>Who, accent, texture and room live in the speaker embedding. Switch to Designable to steer them; your moves here are kept.</span>
    <button class="btn sm" onClick={() => setMode('embedding')}>
      Switch to Designable
    </button>
  </>
);

export function CastScreen() {
  useEffect(() => {
    activeHistory.value = designs as History<unknown>;
    primaryAction.value = togglePreview;
    return () => {
      activeHistory.value = null;
      primaryAction.value = null;
    };
  }, []);
  const b = designBinding();
  const locked = design.value.mode === 'reference' ? CLONE_LOCK : undefined;
  const [open, setOpen] = useState<{ id: CardId; card: HTMLElement; returnTo: HTMLElement } | null>(null);
  const expand = (id: CardId) => (card: HTMLElement, returnTo: HTMLElement) => setOpen({ id, card, returnTo });
  // A card that locks (Exact clone) closes its dialog.
  useEffect(() => {
    if (open && locked && (open.id === 'texture' || open.id === 'room')) setOpen(null);
  }, [!!locked]);
  return (
    <div class="cast">
      <LibraryRail />
      <section class="bench" aria-label="Voice bench">
        <SpeakPanel />
        <BenchHeader />
        <div class="hero">
          <Panel family="who" title="Who" locked={locked} aside={<span class="hint" title="Drag the dot to move gender / age, scroll for pitch, click a voice to start from it">drag · scroll = pitch</span>}>
            <WhoMap />
          </Panel>
          <Panel family="accent" title="Accent" locked={locked}>
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
      </section>
      <aside class="monitor" aria-label="Monitor">
        <div class="monitor-scroll">
          <PreviewSettings />
          <Personas />
          <Variations />
          <SavePanel />
        </div>
      </aside>
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
          {open.id === 'texture' ? (
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
