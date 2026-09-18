// Focused image-deletion acceptance check. Uses isolated synthetic data only.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'test-results', `delete-${Date.now()}`);
await mkdir(output, { recursive: true });
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const base = `http://127.0.0.1:${port}`;
const python = path.join(repo, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const server = spawn(python, ['-m', 'uvicorn', 'grounding.api:app', '--host', '127.0.0.1', '--port', String(port)], {
  cwd: repo, env: { ...process.env, GROUNDING_DATA_DIR: path.join(output, 'data') }, windowsHide: true,
});
let log = '', browser, page;
server.stdout.on('data', data => { log += data; });
server.stderr.on('data', data => { log += data; });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function api(route, body) {
  const response = await fetch(`${base}/api${route}`, { method: body ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  assert(response.ok, `${route}: ${await response.clone().text()}`);
  return response.json();
}
try {
  let ready = false;
  for (let i = 0; i < 200; i++) {
    try { ready = (await fetch(`${base}/api/health`)).ok; } catch {}
    if (ready) break;
    await pause(200);
  }
  assert(ready, 'Isolated test API starts');
  await api('/synthetic', { count: 6, seed: 42, reviewed: true });
  const initial = await api('/images');
  const image = initial[0];
  const version = await api('/versions', { name: 'Deletion preservation check' });
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(base);
  await page.locator('nav').getByRole('button', { name: /^Images & annotations/ }).click();
  const remove = page.getByRole('button', { name: `Delete ${image.filename}`, exact: true });
  const dialog = page.getByRole('dialog', { name: 'Delete image?', exact: true });
  let deleteRequests = 0;
  page.on('request', request => { if (request.method() === 'DELETE') deleteRequests++; });
  await remove.click();
  await dialog.waitFor();
  assert(await dialog.getByRole('button', { name: 'Cancel', exact: true }).evaluate(element => element === document.activeElement));
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'detached' });
  await remove.click();
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.equal(deleteRequests, 0, 'Cancel/Escape must not issue DELETE');
  assert.equal((await api('/images')).length, 6);
  await page.getByRole('button', { name: `Open ${image.filename}`, exact: true }).click();
  await page.getByRole('img', { name: 'Interactive screenshot annotation canvas' }).waitFor();
  await page.getByTitle('Close workspace', { exact: true }).click();
  await remove.click();
  const endpoint = `**/api/images/${image.id}`;
  await page.route(endpoint, route => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Simulated delete failure' }) }));
  await dialog.getByRole('button', { name: 'Delete image', exact: true }).click();
  await dialog.getByRole('alert').getByText('Simulated delete failure').waitFor();
  assert.equal((await api('/images')).length, 6, 'Failure preserves image');
  await page.unroute(endpoint);
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route(endpoint, async route => { await gate; await route.continue(); });
  await dialog.getByRole('button', { name: 'Delete image', exact: true }).click();
  assert(await dialog.getByRole('button', { name: 'Deleting', exact: true }).isDisabled(), 'Block repeat submissions');
  assert(await dialog.getByRole('button', { name: 'Cancel', exact: true }).isDisabled());
  release();
  await dialog.waitFor({ state: 'detached' });
  await remove.waitFor({ state: 'detached' });
  const remaining = await api('/images');
  assert.deepEqual(new Set(remaining.map(im => im.id)), new Set(initial.slice(1).map(im => im.id)));
  assert.equal((await api('/dashboard')).images, 5);
  assert.equal(await page.locator('.image-card').count(), 5);
  const preserved = await api(`/versions/${version.id}`);
  assert.deepEqual(preserved, version);
  const archive = await fetch(`${base}/api/versions/${version.id}/export`);
  assert.equal(archive.status, 200, 'Snapshot image hashes still verify after deleting live image');
  assert((await archive.arrayBuffer()).byteLength > 1000);
  assert.deepEqual(errors, []);
  await page.screenshot({ path: path.join(output, 'after-deletion.png'), fullPage: true });
  console.log('PASS: cancel/Escape, normal annotation opening, failure recovery, duplicate-submit protection, delete/count refresh, and immutable snapshot preservation.');
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, remaining_images: 5, snapshot_examples: preserved.records.length, browser_errors: errors }, null, 2));
} catch (error) {
  if (page) await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill();
  await writeFile(path.join(output, 'server.log'), log);
}
