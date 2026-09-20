// Teach chat to an actual saved grounder and verify both tasks from the same run.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'test-results', `chat-${Date.now()}`);
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
let log = '', browser, page, runId, groundingRunId;
const errors = [];
server.stdout.on('data', data => { log += data; });
server.stderr.on('data', data => { log += data; });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function api(route, body) {
  const response = await fetch(`${base}/api${route}`, { method: body === undefined ? 'GET' : 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  assert(response.ok, `${route}: ${response.status} ${await response.clone().text()}`);
  return response.json();
}
async function groundingPrediction(id, screenshot, instruction) {
  const form = new FormData();
  form.set('file', new Blob([screenshot], { type: 'image/png' }), 'screen.png');
  form.set('run_id', id);
  form.set('instruction', instruction);
  form.set('checkpoint', 'latest');
  const response = await fetch(`${base}/api/predict`, { method: 'POST', body: form });
  assert(response.ok, `grounding prediction: ${response.status} ${await response.clone().text()}`);
  return response.json();
}
async function until(fn, timeout = 90000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const result = await fn();
    if (result) return result;
    await pause(250);
  }
  throw new Error('Timed out waiting for chat workflow');
}
try {
  await until(async () => { try { return (await fetch(`${base}/api/health`)).ok; } catch { return false; } });
  await api('/synthetic', { count: 6, seed: 6, reviewed: true });
  const version = await api('/versions', { name: 'Grounding source', seed: 4, train_ratio: .67, val_ratio: .17 });
  const defaults = await api('/training/defaults');
  const sourceRun = await api('/runs', { name: 'Saved grounding model', version_id: version.id,
    config: { ...defaults, image_size: 32, batch_size: 3, epochs: 1, width: 8, text_dim: 64, max_tokens: 16,
      device: 'cpu', checkpoint_every: 1 } });
  groundingRunId = sourceRun.id;
  await until(async () => {
    const current = await api(`/runs/${groundingRunId}`);
    assert.notEqual(current.status, 'error', current.error);
    return current.status === 'completed';
  });
  const sourceImage = (await api('/images'))[0];
  const imageResponse = await fetch(`${base}${sourceImage.url}`);
  assert(imageResponse.ok);
  const screenshot = await imageResponse.arrayBuffer();
  const instruction = sourceImage.examples[0].instruction;
  const originalGrounding = await groundingPrediction(groundingRunId, screenshot, instruction);
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(20000);
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(base);
  await page.locator('nav').getByRole('button', { name: 'Training', exact: true }).click();
  await page.getByRole('button', { name: 'Chat', exact: true }).click();
  await page.getByLabel('Grounding model to train for chat', { exact: true }).selectOption(`grounding:${groundingRunId}`);
  await page.getByLabel('User message', { exact: true }).fill('hey');
  await page.getByLabel('Your preferred reply', { exact: true }).fill('Hello!');
  await page.getByRole('button', { name: 'Save example', exact: true }).click();
  await until(async () => (await api('/chat/examples')).length === 1);
  await page.getByLabel('User message', { exact: true }).fill('bye');
  await page.getByLabel('Your preferred reply', { exact: true }).fill('Goodbye!');
  await page.getByRole('button', { name: 'Save example', exact: true }).click();
  await until(async () => (await api('/chat/examples')).length === 2);
  await page.getByLabel('Chat experiment name', { exact: true }).fill('Greeting smoke');
  await page.getByLabel('Epochs', { exact: true }).fill('200');
  await page.getByText('Training settings', { exact: true }).click();
  await page.getByLabel('Compute device', { exact: true }).selectOption('cpu');
  await page.getByRole('button', { name: 'Train chat model', exact: true }).click();
  const run = await until(async () => (await api('/chat/runs'))[0]);
  runId = run.id;
  assert.equal(run.source_run_id, groundingRunId);
  await page.getByRole('button', { name: 'Stop & save', exact: true }).click();
  const stopped = await until(async () => {
    const current = await api(`/chat/runs/${runId}`);
    assert.notEqual(current.status, 'error', current.error);
    return current.status === 'stopped' && current;
  });
  assert(stopped.progress.epoch < 200, 'Stop must preserve unfinished work');
  await page.getByRole('button', { name: 'Resume run', exact: true }).click();
  const completed = await until(async () => {
    const current = await api(`/chat/runs/${runId}`);
    assert.notEqual(current.status, 'error', current.error);
    return current.status === 'completed' && current;
  }, 180000);
  assert(completed.progress.global_step > 0);
  await until(async () => (await page.getByLabel('Chat model', { exact: true }).locator('option:checked').textContent())?.includes('completed'));
  await page.getByLabel('Test message', { exact: true }).fill('hey');
  const predictionPromise = page.waitForResponse(response => response.url().endsWith('/api/chat/predict') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  const predictionResponse = await predictionPromise;
  assert(predictionResponse.ok(), await predictionResponse.text());
  assert.equal((await predictionResponse.json()).reply, 'Hello!');
  await page.locator('.chat-message.model').filter({ hasText: 'Hello!' }).waitFor();
  const bye = await api('/chat/predict', { run_id: runId, message: 'bye' });
  assert.equal(bye.reply, 'Goodbye!');
  const unifiedGrounding = await groundingPrediction(runId, screenshot, instruction);
  for (const key of ['raw_bbox', 'presence_score', 'raw_class_id', 'target_present', 'bbox', 'click_point']) {
    assert.deepEqual(unifiedGrounding[key], originalGrounding[key], `Chat training changed grounding output ${key}`);
  }
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download grounding + chat model', exact: true }).click();
  const download = await downloadPromise;
  assert.equal(await download.failure(), null);
  assert(download.suggestedFilename().endsWith('.pt'));
  await download.saveAs(path.join(output, 'grounding-chat-model.pt'));
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, 'chat-desktop.png'), fullPage: true });
  await page.locator('nav').getByRole('button', { name: 'Prediction', exact: true }).click();
  await page.getByLabel('Trained model', { exact: true }).selectOption(runId);
  await page.getByLabel('Instruction', { exact: true }).fill(instruction);
  await page.locator('input[type=file]').setInputFiles({ name: 'same-model.png', mimeType: 'image/png', buffer: Buffer.from(screenshot) });
  const groundingResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/predict') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Find target', exact: true }).click();
  const groundingResponse = await groundingResponsePromise;
  assert(groundingResponse.ok(), await groundingResponse.text());
  const uiGrounding = await groundingResponse.json();
  assert.equal(uiGrounding.run_id, runId);
  assert.deepEqual(uiGrounding.raw_bbox, originalGrounding.raw_bbox);
  await page.getByRole('img', { name: 'Screenshot with normalized target overlays' }).waitFor();
  await page.screenshot({ path: path.join(output, 'same-model-prediction.png'), fullPage: true });
  await page.locator('nav').getByRole('button', { name: 'Training', exact: true }).click();
  await page.reload();
  await page.getByRole('button', { name: 'Chat', exact: true }).click();
  assert.equal((await api('/chat/examples')).length, 2);
  await page.getByText('Greeting smoke', { exact: true }).first().waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await until(async () => page.locator('.sidebar').evaluate(element => element.getBoundingClientRect().right <= 0));
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile chat view must not overflow horizontally');
  await page.screenshot({ path: path.join(output, 'chat-mobile.png'), fullPage: true });
  assert.deepEqual(errors, []);
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, grounding_run_id: groundingRunId,
    unified_run_id: runId, epochs: completed.progress.epoch, loss: completed.progress.loss,
    replies: ['Hello!', 'Goodbye!'], grounding_unchanged: true, browser_errors: errors }, null, 2));
  console.log(`PASS: saved grounding model selected, real chat training stopped/resumed, both learned replies, identical grounding predictions, unified download, persistence and mobile layout. Evidence: ${output}`);
} catch (error) {
  if (page) await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
  throw error;
} finally {
  if (runId) {
    try {
      const run = await api(`/chat/runs/${runId}`);
      if (['queued', 'running'].includes(run.status)) {
        await api(`/chat/runs/${runId}/stop`, {});
        await until(async () => !['queued', 'running'].includes((await api(`/chat/runs/${runId}`)).status), 30000);
      }
    } catch { /* preserve original failure and server log */ }
  }
  if (groundingRunId) {
    try {
      const current = await api(`/runs/${groundingRunId}`);
      if (['queued', 'running'].includes(current.status)) {
        await api(`/runs/${groundingRunId}/stop`, {});
        await until(async () => !['queued', 'running'].includes((await api(`/runs/${groundingRunId}`)).status), 30000);
      }
    } catch { /* preserve original failure and server log */ }
  }
  if (browser) await browser.close();
  server.kill();
  await writeFile(path.join(output, 'server.log'), log);
}
