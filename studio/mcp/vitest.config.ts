import { defineConfig } from 'vitest/config';

// vite.config.ts roots the widget build in widget/; the tests run from the package.
export default defineConfig({
  test: { root: '.', include: ['tests/**/*.test.ts'], environment: 'node' },
});
