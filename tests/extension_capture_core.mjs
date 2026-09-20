import test from 'node:test';
import assert from 'node:assert/strict';
import { runCapture, validateSettings } from '../capture-extension/capture-core.js';

const settings = { tabId: 12, name: 'Responsive views', group: 'same-page', delayMs: 250,
  resolutions: [{ width: 390, height: 844 }, { width: 1280, height: 720 }] };

// The fake CDP adapter only needs IHDR bytes; the installed-extension test verifies complete real PNGs.
function screenshotHeader(width, height) {
  const bytes = Buffer.alloc(24);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes);
  bytes.writeUInt32BE(13, 8); bytes.write('IHDR', 12);
  bytes.writeUInt32BE(width, 16); bytes.writeUInt32BE(height, 20);
  return bytes.toString('base64');
}

function fixture() {
  const calls = [], saved = [], images = new Map();
  let current = { clientWidth: 900, clientHeight: 700, pageX: 4, pageY: 217 }, number = 0;
  const adapter = {
    async attach(tabId) { calls.push({ method: 'attach', tabId }); },
    async detach(tabId) { calls.push({ method: 'detach', tabId }); },
    async send(tabId, method, params) {
      calls.push({ method, params, tabId });
      if (method === 'Emulation.setDeviceMetricsOverride') current = { clientWidth: params.width, clientHeight: params.height, pageX: 0, pageY: 0 };
      if (method === 'Page.getLayoutMetrics') return { cssLayoutViewport: { ...current } };
      if (method === 'Page.captureScreenshot') return { data: screenshotHeader(current.clientWidth, current.clientHeight) };
      return {};
    },
    async sleep() {},
    uuid: () => `capture-${++number}`,
    now: () => '2026-09-20T00:00:00.000Z',
  };
  const store = {
    async saveBatch(batch) { saved.push(structuredClone(batch)); },
    async saveImage(id, data) { images.set(id, data); },
  };
  const batch = { id: 'test-batch', status: 'capturing', source_url: 'http://localhost/fixture',
    title: 'Responsive fixture', group: settings.group, captures: [], errors: [] };
  return { adapter, store, batch, calls, saved, images };
}

function assertRestored(calls) {
  const clearing = calls.findIndex(call => call.method === 'Emulation.clearDeviceMetricsOverride');
  assert(clearing >= 0);
  const restore = calls[clearing + 1];
  assert.equal(restore.method, 'Runtime.evaluate');
  assert.match(restore.params.expression, /left:4,top:217/);
  assert.equal(calls.at(-1).method, 'detach');
}

test('settings deduplicate sizes but enforce bounded viewport and batch work', () => {
  assert.deepEqual(validateSettings({ ...settings, resolutions: [...settings.resolutions, settings.resolutions[0]] }).resolutions,
    settings.resolutions);
  for (const changes of [
    { tabId: 0 }, { delayMs: 0 }, { resolutions: [] },
    { resolutions: [{ width: 239, height: 480 }] },
    { resolutions: [{ width: 4097, height: 480 }] },
    { resolutions: [{ width: 390.5, height: 844 }] },
    { resolutions: Array.from({ length: 9 }, (_, index) => ({ width: 320 + index, height: 480 })) },
    { resolutions: [{ width: 4096, height: 4096 }, { width: 4095, height: 4096 }, { width: 4094, height: 4096 }] },
  ]) assert.throws(() => validateSettings({ ...settings, ...changes }));
});

test('capture applies actual metrics for each screenshot and always restores the tab', async () => {
  const value = fixture();
  const result = await runCapture({ ...value, settings });
  assert.equal(result.status, 'completed');
  assert.equal(value.images.size, 2);
  assert.deepEqual(result.captures.map(row => [row.width, row.height]), [[390, 844], [1280, 720]]);
  assert.deepEqual(result.original_scroll, { x: 4, y: 217 });
  assert(result.captures.every(row => row.url === value.batch.source_url && row.device_scale_factor === 1));
  const screenshots = value.calls.filter(call => call.method === 'Page.captureScreenshot');
  assert.deepEqual(screenshots.map(call => call.params.clip), [
    { x: 0, y: 0, width: 390, height: 844, scale: 1 },
    { x: 0, y: 0, width: 1280, height: 720, scale: 1 },
  ]);
  assert(screenshots.every(call => call.params.captureBeyondViewport === false));
  assertRestored(value.calls);
  assert.equal(value.saved.at(-1).status, 'completed');
});

test('cancel preserves completed images, restores the original page, and releases the debugger', async () => {
  const value = fixture();
  const controller = new AbortController();
  const result = await runCapture({ ...value, settings, signal: controller.signal,
    onUpdate(batch) {
      if (batch.captures.length === 1 && !controller.signal.aborted) controller.abort(new DOMException('Canceled by user', 'AbortError'));
    },
  });
  assert.equal(result.status, 'canceled');
  assert.equal(result.captures.length, 1);
  assert.equal(value.images.size, 1);
  assert.equal(value.calls.filter(call => call.method === 'Page.captureScreenshot').length, 1);
  assertRestored(value.calls);
  assert.equal(value.saved.at(-1).status, 'canceled');
});

test('a screenshot failure keeps prior captures and performs cleanup', async () => {
  const value = fixture();
  const originalSend = value.adapter.send;
  let screenshots = 0;
  value.adapter.send = async (tabId, method, params) => {
    if (method === 'Page.captureScreenshot' && ++screenshots === 2) throw new Error('Tab screenshot failed');
    return originalSend(tabId, method, params);
  };
  const result = await runCapture({ ...value, settings });
  assert.equal(result.status, 'error');
  assert.equal(result.captures.length, 1);
  assert.match(result.errors.join(' '), /Tab screenshot failed/);
  assertRestored(value.calls);
});

test('a failed viewport restore still restores scroll and detaches', async () => {
  const value = fixture();
  const originalSend = value.adapter.send;
  value.adapter.send = async (tabId, method, params) => {
    const result = await originalSend(tabId, method, params);
    if (method === 'Emulation.clearDeviceMetricsOverride') throw new Error('Lost viewport session');
    return result;
  };
  const result = await runCapture({ ...value, settings });
  assert.equal(result.status, 'error');
  assert.equal(result.captures.length, 2);
  assert.match(result.errors.join(' '), /Restore viewport/);
  assertRestored(value.calls);
});
