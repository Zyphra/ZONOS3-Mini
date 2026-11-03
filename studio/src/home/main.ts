/**
 * The chooser at `/`: the Voice Studio (`/studio`) or the lite voice-cloning page (`/lite`). The page's query
 * (`?api=web`, `?server=…`) carries over to whichever one opens.
 */
import '@fontsource-variable/inter/wght.css';
import './home.css';
import { mountChrome } from '../lib/chrome';

mountChrome();
if (location.search) for (const a of document.querySelectorAll<HTMLAnchorElement>('a[data-keep-query]')) a.search = location.search;
