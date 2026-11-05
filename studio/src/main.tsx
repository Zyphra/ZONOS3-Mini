import { render } from 'preact';
import { App } from './app';
import '@fontsource-variable/inter/wght.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import '@fontsource/jetbrains-mono/latin-500.css';
import './state/theme';
import './styles.css';
import './styles/mobile.css';

render(<App />, document.getElementById('app')!);

// The build's /sw.js (vite.config.ts serviceWorker()); the dev server has none. A failed registration leaves the app
// as it is: online only.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {});
  });
}
