/** The phone layout (portrait phones); the same breakpoint as the `@media (max-width: 760px)` rules in styles/mobile.css. */
import { signal } from '@preact/signals';

const query = matchMedia('(max-width: 760px)');

export const phone = signal(query.matches);
query.addEventListener('change', (e) => (phone.value = e.matches));
