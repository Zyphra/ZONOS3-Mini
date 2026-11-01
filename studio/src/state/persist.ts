/**
 * Browser-wide preferences in localStorage (theme, autoplay, edit / dub options). Everything a project holds lives
 * in the project store (`projects.ts`, IndexedDB).
 */
import { effect, type Signal } from '@preact/signals';

const PREFIX = 'zvs.';

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

/** Mirror a signal into localStorage (debounced). */
export function persist<T>(key: string, s: Signal<T>) {
  let timer = 0;
  effect(() => {
    const v = s.value;
    clearTimeout(timer);
    timer = window.setTimeout(() => {
      try {
        localStorage.setItem(PREFIX + key, JSON.stringify(v));
      } catch {
        /* quota or privacy mode: the session still works */
      }
    }, 300);
  });
}
