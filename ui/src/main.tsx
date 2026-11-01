import { render } from 'preact';
import { App } from './app';
import '@fontsource-variable/inter/wght.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import '@fontsource/jetbrains-mono/latin-500.css';
import './state/theme';
import './styles.css';

render(<App />, document.getElementById('app')!);
