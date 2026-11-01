import { signal, type Signal } from '@preact/signals';

/**
 * Undo/redo over immutable snapshots. A drag is one gesture: `begin()` remembers the state, `set()` updates
 * it live, `end()` records one undo step if anything changed. `apply()` is a one-shot gesture.
 */
export class History<T> {
  readonly state: Signal<T>;
  readonly canUndo = signal(false);
  readonly canRedo = signal(false);
  private past: T[] = [];
  private future: T[] = [];
  private gesture: T | null = null;

  constructor(
    initial: T,
    private readonly limit = 200,
  ) {
    this.state = signal(initial);
  }

  begin() {
    if (this.gesture === null) this.gesture = this.state.value;
  }

  set(next: T) {
    this.state.value = next;
  }

  end() {
    const before = this.gesture;
    this.gesture = null;
    if (before === null || before === this.state.value) return;
    this.past.push(before);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
    this.sync();
  }

  apply(next: T) {
    this.begin();
    this.set(next);
    this.end();
  }

  undo() {
    this.end();
    const prev = this.past.pop();
    if (prev === undefined) return;
    this.future.push(this.state.value);
    this.state.value = prev;
    this.sync();
  }

  redo() {
    this.end();
    const next = this.future.pop();
    if (next === undefined) return;
    this.past.push(this.state.value);
    this.state.value = next;
    this.sync();
  }

  /** Replace the state without an undo step (loading, migrations). */
  reset(next: T) {
    this.past = [];
    this.future = [];
    this.gesture = null;
    this.state.value = next;
    this.sync();
  }

  private sync() {
    this.canUndo.value = this.past.length > 0;
    this.canRedo.value = this.future.length > 0;
  }
}
