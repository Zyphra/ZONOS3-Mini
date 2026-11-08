import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';

export default defineConfig({
  plugins: [preact()],
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'happy-dom',
    setupFiles: ['src/test/setup.ts'],
  },
});
