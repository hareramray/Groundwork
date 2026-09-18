// Automated tests against this repository's own localhost application, in an isolated browser/data folder.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'test-results', `ui-${Date.now()}`);
await mkdir(output, { recursive: true });
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const base = `http://127.0.0.1:${port}`;
const python = process.platform === 'win32' ? path.join(repo, '.venv', 'Scripts', 'python.exe') : path.join(repo, '.venv', 'bin', 'python');
const server = spawn(python, ['-m', 'uvicorn', 'grounding.api:app', '--host', '127.0.0.1', '--port', String(port)], {
  cwd: repo, env: { ...process.env, GROUNDING_DATA_DIR: path.join(output, 'data') }, windowsHide: true,
});
let log = '', browser, page;
server.stdout.on('data', data => { log += data; });
server.stderr.on('data', data => { log += data; });
const evidence = [];
const record = message => { evidence.push(message); console.log(message); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function api(route, payload, method = payload === undefined ? 'GET' : 'POST') {
  const response = await fetch(`${base}/api${route}`, { method, headers: { 'Content-Type': 'application/json' }, body: payload === undefined ? undefined : JSON.stringify(payload) });
  assert(response.ok, `${method} ${route}: ${response.status} ${await response.clone().text()}`);
  return response.json();
}
async function until(fn, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await fn(); if (value) return value; await pause(200); }
  throw new Error('Timed out waiting for workflow state');
}
try {
  await until(async () => { try { return (await fetch(`${base}/api/health`)).ok; } catch { return false; } }, 45000);
  await api('/synthetic', { count: 6, seed: 42, reviewed: true });
  const image = (await api('/images'))[0];
  const pixels = await readFile(path.join(output, 'data', image.image_path));
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.setDefaultTimeout(15000);
  await page.goto(base);
  await page.getByRole('heading', { name: 'A small model. A clear purpose.' }).waitFor();
  await page.screenshot({ path: path.join(output, 'dashboard.png'), fullPage: true });
  record('Dashboard renders real local hardware and dataset counts.');
  await page.locator('nav').getByRole('button', { name: /^Images & annotations/ }).click();
  await page.locator('input[type=file]').setInputFiles({ name: 'annotation-test.png', mimeType: 'image/png', buffer: pixels });
  await page.locator('.image-card').filter({ hasText: 'annotation-test.png' }).click();
  const canvas = page.getByRole('img', { name: 'Interactive screenshot annotation canvas' });
  await canvas.waitFor();
  async function screenPoint(x, y) {
    return canvas.evaluate((svg, p) => { const g = svg.querySelector('g'); const point = new DOMPoint(p[0] * 640, p[1] * 400).matrixTransform(g.getScreenCTM()); return [point.x, point.y]; }, [x, y]);
  }
  async function drag(a, b) {
    const start = await screenPoint(...a), end = await screenPoint(...b);
    await page.mouse.move(...start); await page.mouse.down(); await page.mouse.move(...end, { steps: 6 }); await page.mouse.up();
  }
  await page.getByTitle('Draw target (D)', { exact: true }).click();
  await drag([.12, .2], [.38, .37]);
  await page.getByLabel('Visible label (optional)', { exact: true }).fill('Review search');
  await page.getByLabel('Website / template / session group', { exact: false }).fill('ui-test-group');
  await page.getByRole('button', { name: 'Add instruction for this target', exact: true }).click();
  await page.getByLabel('Instruction', { exact: true }).fill('Find the review search field');
  await page.getByLabel('Review status', { exact: false }).selectOption('reviewed');
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const annotated = await until(async () => { const item = (await api('/images')).find(im => im.filename === 'annotation-test.png'); return item?.examples.length === 1 && item; });
  assert.equal(annotated.status, 'reviewed');
  annotated.elements[0].bbox.forEach((value, i) => assert(Math.abs(value - [.12, .2, .38, .37][i]) < .005));
  record('Upload, draw box, label target, link instruction, explicit review and persistent save pass.');
  // Zoom and drag using DOM CTM proves inverse normalization under the actual rendered transform.
  await page.getByTitle('Zoom in', { exact: true }).click();
  await page.getByTitle('Zoom in', { exact: true }).click();
  await page.getByRole('button', { name: /^1\. Elements/ }).click();
  await drag([.18, .25], [.23, .29]);
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const moved = await until(async () => { const item = await api(`/images/${annotated.id}`); return item.elements[0].bbox[0] > .15 && item; });
  const expected = [.17, .24, .43, .41];
  moved.elements[0].bbox.forEach((value, i) => assert(Math.abs(value - expected[i]) < .006));
  assert.equal(moved.examples[0].status, 'draft', 'Editing a reviewed target must require review again');
  await drag([moved.elements[0].bbox[2], moved.elements[0].bbox[3]], [.48, .46]);
  await page.getByTitle('Adjust click point', { exact: true }).click();
  await page.mouse.click(...await screenPoint(.30, .35));
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const resized = await until(async () => { const item = await api(`/images/${annotated.id}`); return item.elements[0].bbox[2] > .46 && item; });
  assert(Math.abs(resized.elements[0].bbox[3] - .46) < .006);
  assert(Math.abs(resized.elements[0].click_point[0] - .30) < .006);
  assert(Math.abs(resized.elements[0].click_point[1] - .35) < .006);
  await page.getByTitle('Draw target (D)', { exact: true }).click();
  await drag([.65, .50], [.8, .65]);
  await page.getByTitle('Delete selected element', { exact: true }).click();
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  assert.equal((await api(`/images/${annotated.id}`)).elements.length, 1);
  await page.setViewportSize({ width: 1100, height: 850 });
  await page.screenshot({ path: path.join(output, 'annotation.png'), fullPage: true });
  record('Zoomed move/resize/click-point adjustment/delete preserve coordinates; edited targets return to draft; responsive canvas renders.');
  await page.getByTitle('Close workspace', { exact: true }).click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole('button', { name: 'Dataset versions', exact: true }).click();
  await page.getByLabel('Version name', { exact: true }).fill('Browser UI verification');
  await page.getByRole('button', { name: 'Validate annotations', exact: true }).click();
  await page.getByText('Validation passed', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Create version', exact: true }).click();
  const version = await until(async () => (await api('/versions'))[0]);
  assert.equal(version.stats.examples, 30, 'Edited draft excluded from immutable snapshot');
  record('Dataset UI validation/versioning passes and excludes draft edits.');
  await page.getByRole('button', { name: 'Training', exact: true }).click();
  await page.getByLabel('Experiment name', { exact: true }).fill('Browser UI tiny model');
  await page.getByLabel('Image resolution (square)', { exact: true }).fill('64');
  await page.getByLabel('Batch size', { exact: true }).fill('2');
  await page.getByLabel('Epochs', { exact: true }).fill('1');
  await page.getByLabel('Architecture preset', { exact: false }).selectOption('16-32');
  await page.getByRole('button', { name: 'Start fresh training', exact: true }).click();
  await page.getByRole('button', { name: 'Pause & save', exact: true }).click();
  const paused = await until(async () => { const current = (await api('/runs'))[0]; if (current?.status === 'error') throw new Error(current.error); return current?.status === 'paused' && current; }, 120000);
  assert(paused.latest_checkpoint);
  await page.getByRole('button', { name: 'Resume this exact run', exact: true }).click();
  const run = await until(async () => { const current = (await api('/runs'))[0]; if (current?.status === 'error') throw new Error(current.error); return current?.status === 'completed' && current; }, 120000);
  assert(run.latest_checkpoint);
  record('Training UI starts, pauses, saves, resumes a real GPU worker and obtains a completed checkpoint.');
  await page.getByRole('button', { name: 'Models & checkpoints', exact: true }).click();
  await page.getByRole('button', { name: 'Create inference export', exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download', exact: true }).click();
  const download = await downloadPromise;
  await download.saveAs(path.join(output, 'export.pt'));
  assert((await readFile(path.join(output, 'export.pt'))).length > 1000);
  record('Model library lists saved checkpoints and creates/downloads a real inference export.');
  await page.getByRole('button', { name: 'Retrain with new data', exact: true }).click();
  await until(async () => await page.getByLabel('Architecture preset', { exact: false }).inputValue() === '16-32');
  assert(await page.getByLabel('Architecture preset', { exact: false }).isDisabled());
  await page.getByLabel('Experiment name', { exact: true }).fill('Browser UI child');
  await page.getByLabel('Image resolution (square)', { exact: true }).fill('64');
  await page.getByLabel('Epochs', { exact: true }).fill('1');
  await page.getByRole('button', { name: 'Start retraining run', exact: true }).click();
  const child = await until(async () => { const current = (await api('/runs'))[0]; if (current?.status === 'error') throw new Error(current.error); return current?.name === 'Browser UI child' && current.status === 'completed' && current; }, 120000);
  assert.equal(child.parent_run_id, run.id);
  assert.equal(child.config.width, 16);
  record('Retraining UI inherits the parent architecture and creates a distinct child run.');
  await page.getByRole('button', { name: 'Prediction', exact: true }).click();
  await page.getByLabel('Instruction', { exact: true }).fill('Find the blue button');
  await page.locator('input[type=file]').setInputFiles({ name: 'prediction-test.png', mimeType: 'image/png', buffer: pixels });
  await page.getByRole('button', { name: 'Find target', exact: true }).click();
  await page.getByRole('button', { name: 'Prediction JSON', exact: true }).waitFor({ timeout: 30000 });
  await page.getByLabel('Instruction', { exact: true }).fill('Find the green textbox');
  await page.getByRole('button', { name: 'Prediction JSON', exact: true }).waitFor({ state: 'detached' });
  await page.getByRole('button', { name: 'Find target', exact: true }).click();
  await page.getByRole('button', { name: 'Prediction JSON', exact: true }).waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Correct prediction', exact: true }).click();
  await page.getByLabel('Target visibility', { exact: false }).selectOption('absent');
  await page.getByRole('button', { name: 'Save draft & open annotation workspace', exact: true }).click();
  await canvas.waitFor();
  const correction = (await api('/images')).find(im => im.filename === 'prediction-test.png');
  assert.equal(correction.examples[0].status, 'draft');
  assert.equal(correction.examples[0].target_present, false);
  record('Prediction, score display, correction and mandatory draft review work through the UI.');
  await page.getByTitle('Close workspace', { exact: true }).click();
  await page.getByRole('button', { name: 'Evaluation', exact: true }).click();
  await page.getByRole('button', { name: 'Run evaluation', exact: true }).click();
  await page.getByRole('heading', { name: 'Measured results', exact: true }).waitFor({ timeout: 30000 });
  await page.screenshot({ path: path.join(output, 'evaluation.png'), fullPage: true });
  assert.equal(errors.length, 0, errors.join('\n'));
  record('Evaluation renders actual metrics/baselines/overlays; no browser page errors.');
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, evidence, errors }, null, 2));
  console.log(`PASS: ${output}`);
} catch (error) {
  if (page) await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: false, evidence, error: String(error) }, null, 2));
  throw error;
} finally {
  if (browser) await browser.close();
  try {
    for (const run of await api('/runs')) if (['running', 'queued'].includes(run.status)) await api(`/runs/${run.id}/stop`, {});
  } catch {}
  server.kill();
  await writeFile(path.join(output, 'server.log'), log);
}
