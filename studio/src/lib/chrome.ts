/**
 * The studio's top-bar chrome as plain DOM, for the pages without Preact (the chooser at `/`, the lite page): the
 * brand marks, `<Icon>`-style labels and the theme toggle (`localStorage['zvs.theme']`, shared with the studio; each
 * page's HTML applies it before first paint).
 */
import { FILLED_ICONS, ICONS, MARK, MARK_VIEWBOX, WORD, WORD_VIEWBOX } from './marks';

const SVG_NS = 'http://www.w3.org/2000/svg';

function svg(cls: string, viewBox: string, paths: string[], width: number | null, height: number, stroke = false): SVGSVGElement {
  const el = document.createElementNS(SVG_NS, 'svg');
  el.setAttribute('class', cls);
  el.setAttribute('viewBox', viewBox);
  if (width !== null) el.setAttribute('width', String(width));
  el.setAttribute('height', String(height));
  el.setAttribute('aria-hidden', 'true');
  for (const d of paths) {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('d', d);
    if (stroke) {
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', 'currentColor');
      p.setAttribute('stroke-width', '1.8');
      p.setAttribute('stroke-linecap', 'round');
      p.setAttribute('stroke-linejoin', 'round');
    } else p.setAttribute('fill', 'currentColor');
    el.append(p);
  }
  return el;
}

/** A button's text after its icon (`data-icon` unless named): the studio's `<Icon>` as plain DOM. */
export function setLabel(el: HTMLElement, text: string, name = el.dataset.icon, size = 16) {
  el.replaceChildren(...(name ? [svg('icon', '0 0 24 24', [ICONS[name]], size, size, !FILLED_ICONS[name])] : []), text);
}

/**
 * The page chrome: the Zyphra marks into `.brand`, icons into every `[data-icon]` element, and the theme toggle on
 * `#theme` (`onTheme` runs after each switch, e.g. to redraw a canvas).
 */
export function mountChrome(onTheme?: () => void) {
  document.querySelector('.brand')!.prepend(svg('z-mark', MARK_VIEWBOX, MARK, 22, (22 * 164) / 224), svg('z-word', WORD_VIEWBOX, WORD, null, 11));
  for (const el of document.querySelectorAll<HTMLElement>('[data-icon]')) setLabel(el, el.textContent!.trim());
  const button = document.getElementById('theme') as HTMLButtonElement;
  const show = () => {
    const light = document.documentElement.dataset.theme === 'light';
    setLabel(button, '', light ? 'moon' : 'sun');
    button.title = light ? 'Dark theme' : 'Light theme';
    button.setAttribute('aria-label', light ? 'Switch to the dark theme' : 'Switch to the light theme');
  };
  button.addEventListener('click', () => {
    const light = document.documentElement.dataset.theme !== 'light';
    if (light) document.documentElement.dataset.theme = 'light';
    else delete document.documentElement.dataset.theme;
    document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute('content', light ? '#f5f3ee' : '#0a0a0a');
    try {
      localStorage.setItem('zvs.theme', JSON.stringify(light ? 'light' : 'dark'));
    } catch {
      /* storage blocked: the theme holds for this page */
    }
    show();
    onTheme?.();
  });
  show();
}
