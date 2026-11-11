import { describe, expect, it } from 'vitest';
import { voiceServerLanguage } from './voices';

const library = (label: string) => ({ label, is_default: true });

describe('a voice’s render language', () => {
  it('is the server language its library label names', () => {
    expect(voiceServerLanguage(library('Liesel de F 1'), 'en_us')).toBe('de');
    expect(voiceServerLanguage(library('Gaspard fr M 1'), 'en_us')).toBe('fr_fr');
    expect(voiceServerLanguage(library('Meilin zh energetic F 1'), 'en_us')).toBe('cmn');
    expect(voiceServerLanguage(library('Imogen en British F 1'), 'de')).toBe('en_us');
  });

  it('keeps the project’s variant of the same language', () => {
    expect(voiceServerLanguage(library('Imogen en British F 1'), 'en_gb')).toBe('en_gb');
  });

  it('is the project’s for voices that name none, or a language no server language speaks', () => {
    expect(voiceServerLanguage(library('AmericanFemale'), 'de')).toBe('de');
    expect(voiceServerLanguage(library('Ananya hi F 1'), 'en_gb')).toBe('en_gb');
    expect(voiceServerLanguage({ label: 'Max de demo', is_default: false }, 'en_us')).toBe('en_us');
    expect(voiceServerLanguage(undefined, 'ko')).toBe('ko');
  });
});
