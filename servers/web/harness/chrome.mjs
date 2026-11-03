// Headless Chrome with WebGPU (Metal) for the harness and the parity runner.
import { globSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const here = path.dirname(fileURLToPath(import.meta.url));

export function chromePath() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const pattern = process.platform === 'darwin'
    ? 'chrome/*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'
    : 'chrome/*/chrome-*/chrome';
  const found = globSync(pattern, { cwd: path.join(here, '.chrome') }).sort();
  if (!found.length) {
    console.error('harness: set CHROME_PATH or install Chrome into harness/.chrome');
    process.exit(1);
  }
  return path.join(here, '.chrome', found[found.length - 1]);
}

export function launchChrome() {
  return puppeteer.launch({
    executablePath: chromePath(),
    headless: process.env.HARNESS_HEADFUL ? false : true,
    protocolTimeout: 0,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'],
  });
}
