/** The project's language, where lines are rendered (Direct, Scene): text normalization and reading-position spans follow it. */
import { LANG_LABEL } from '../lib/languages';
import { projectSettings } from '../state/app';
import { setSettings } from '../state/projects';

export function LanguageSelect() {
  return (
    <label class="lang-row" title="Lines render in this language (the project's): numbers and abbreviations are read in it, and painted spans follow the reading where the server has alignment heads for it">
      language
      <select class="input" value={projectSettings.value.language} onChange={(e) => setSettings({ language: e.currentTarget.value })} aria-label="Language of the lines">
        {Object.entries(LANG_LABEL).map(([code, label]) => (
          <option key={code} value={code}>
            {label}
          </option>
        ))}
      </select>
    </label>
  );
}
