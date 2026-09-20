// Install the unpacked extension in an isolated Chromium profile and capture only a generated localhost page.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { spawn, spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'test-results', `capture-extension-${Date.now()}`);
await mkdir(output, { recursive: true });
const python = process.platform === 'win32' ? path.join(repo, '.venv', 'Scripts', 'python.exe') : path.join(repo, '.venv', 'bin', 'python');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await fn(); if (value) return value; await pause(100); }
  throw new Error('Timed out waiting for viewport capture workflow');
}
const fixtureHtml = `<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>Responsive capture fixture</title>
<style>html,body{margin:0;min-height:2400px;background:rgb(22,163,74);font:24px Arial;color:white}header{height:80px;background:rgb(37,99,235);padding:0 24px;display:flex;align-items:center;justify-content:space-between}nav{display:flex;gap:30px}.mobile{display:none}h1{margin:64px 24px}.marker{position:absolute;top:1400px;left:24px}@media(max-width:600px){html,body{background:rgb(249,115,22)}header{background:rgb(220,38,38)}nav{display:none}.mobile{display:block}}</style></head>
<body><header><strong>Groundwork fixture</strong><nav><span>Home</span><span>Products</span><span>Account</span></nav><span class="mobile">Menu</span></header><h1>Responsive capture proof</h1><p class="marker">Original scroll restoration marker</p></body></html>`;
const fixture = createHttpServer((request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); response.end(fixtureHtml);
});
await new Promise(resolve => fixture.listen(0, '127.0.0.1', resolve));
const fixtureUrl = `http://127.0.0.1:${fixture.address().port}/responsive`;
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const labPort = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const base = `http://127.0.0.1:${labPort}`;
const server = spawn(python, ['-m', 'uvicorn', 'grounding.api:app', '--host', '127.0.0.1', '--port', String(labPort)], {
  cwd: repo, env: { ...process.env, GROUNDING_DATA_DIR: path.join(output, 'data') }, windowsHide: true,
});
let log = '', context, capturePage;
const browserErrors = [];
server.stdout.on('data', value => { log += value; });
server.stderr.on('data', value => { log += value; });
try {
  await until(async () => { try { return (await fetch(`${base}/api/health`)).ok; } catch { return false; } });
  const extensionPath = path.join(repo, 'capture-extension');
  context = await chromium.launchPersistentContext(path.join(output, 'profile'), {
    channel: 'chromium', headless: true, viewport: null,
    args: [`--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`, '--window-size=960,900'],
  });
  const source = await context.newPage();
  source.on('pageerror', error => browserErrors.push(String(error)));
  await source.goto(fixtureUrl);
  await source.evaluate(() => window.scrollTo(0, 217));
  const original = await source.evaluate(() => ({ width: innerWidth, height: innerHeight, x: scrollX, y: scrollY }));
  assert.equal(original.y, 217);
  const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
  const extensionId = new URL(worker.url()).host;
  const tabId = await worker.evaluate(async url => {
    const tabs = await chrome.tabs.query({});
    return tabs.find(tab => tab.url === url)?.id;
  }, fixtureUrl);
  assert(Number.isInteger(tabId), 'The installed extension must see the generated source tab');
  capturePage = await context.newPage();
  capturePage.setDefaultTimeout(20000);
  capturePage.on('pageerror', error => browserErrors.push(String(error)));
  await capturePage.goto(`chrome-extension://${extensionId}/capture.html?tabId=${tabId}`);
  await capturePage.getByLabel('Source tab', { exact: true }).selectOption(String(tabId));
  await capturePage.getByLabel('Capture name', { exact: true }).fill('Responsive viewport proof');
  await capturePage.getByLabel('Dataset group', { exact: true }).fill('capture-fixture');
  await capturePage.getByLabel('Settle delay (ms)', { exact: true }).fill('250');
  for (const checkbox of await capturePage.getByRole('checkbox').all()) await checkbox.uncheck();
  await capturePage.getByRole('checkbox', { name: /390.*844/ }).check();
  await capturePage.getByLabel('Custom width', { exact: true }).fill('1280');
  await capturePage.getByLabel('Custom height', { exact: true }).fill('720');
  await capturePage.getByRole('button', { name: 'Add size', exact: true }).click();
  await capturePage.getByRole('button', { name: 'Capture selected sizes', exact: true }).click();
  await until(async () => {
    const status = await capturePage.locator('#batch-status').getAttribute('data-status');
    if (status === 'error') throw new Error(await capturePage.locator('#batch-errors').innerText());
    return status === 'completed' && await capturePage.locator('#capture-start').isEnabled();
  });
  const exportButton = capturePage.getByRole('button', { name: 'Export capture ZIP', exact: true });
  await until(async () => await exportButton.count() === 1 && await exportButton.isEnabled());
  const downloadEvent = capturePage.waitForEvent('download');
  await exportButton.click();
  const download = await downloadEvent;
  assert.equal(await download.failure(), null);
  const zipPath = path.join(output, 'responsive-captures.zip');
  await download.saveAs(zipPath);
  const inspection = spawnSync(python, ['-c', `import io,json,sys,zipfile
from PIL import Image
with zipfile.ZipFile(sys.argv[1]) as archive:
    metadata=json.loads(archive.read('manifest.json'))
    captures=[]
    for item in metadata['captures']:
        with Image.open(io.BytesIO(archive.read(item['image_path']))) as image:
            captures.append({'width':image.width,'height':image.height,'pixel':image.convert('RGB').getpixel((5,150))})
    print(json.dumps({'manifest':metadata,'images':captures}))`, zipPath], { cwd: repo, encoding: 'utf8', windowsHide: true });
  assert.equal(inspection.status, 0, inspection.stderr);
  const inspected = JSON.parse(inspection.stdout);
  assert.equal(inspected.manifest.format, 'groundwork-captures');
  assert.equal(inspected.manifest.group, 'capture-fixture');
  assert.equal(inspected.manifest.captures.length, 2);
  assert(inspected.manifest.captures.every(item => item.url === fixtureUrl && item.device_scale_factor === 1));
  assert.deepEqual([...inspected.images].sort((a, b) => a.width - b.width), [
    { width: 390, height: 844, pixel: [249, 115, 22] },
    { width: 1280, height: 720, pixel: [22, 163, 74] },
  ], 'PNG dimensions and CSS breakpoint colors must reflect real viewport changes');
  const restored = await source.evaluate(() => ({ width: innerWidth, height: innerHeight, x: scrollX, y: scrollY }));
  assert.deepEqual(restored, original, 'The original viewport and scroll must be restored');
  assert(await worker.evaluate(async id => {
    await chrome.debugger.attach({ tabId: id }, '1.3');
    await chrome.debugger.detach({ tabId: id });
    return true;
  }, tabId), 'The capture worker must release the debugger');
  await capturePage.screenshot({ path: path.join(output, 'extension-capture.png'), fullPage: true });
  const lab = await context.newPage();
  lab.on('pageerror', error => browserErrors.push(String(error)));
  await lab.goto(base);
  await lab.locator('nav').getByRole('button', { name: /^Images & annotations/ }).click();
  const importResponse = lab.waitForResponse(response => response.url().endsWith('/api/captures/import'));
  await lab.locator('input[aria-label="Import capture ZIP"]').setInputFiles(zipPath);
  const imported = await importResponse;
  assert(imported.ok(), await imported.text());
  assert.equal((await imported.json()).images, 2);
  const imagesResponse = await fetch(`${base}/api/images`);
  const images = await imagesResponse.json();
  assert.equal(images.length, 2);
  assert(images.every(image => image.status === 'unannotated' && image.group === 'capture-fixture' && image.examples.length === 0 && image.elements.length === 0));
  await until(async () => await lab.locator('.image-card').count() === 2);
  await lab.locator('.image-card img').evaluateAll(images => Promise.all(images.map(image => image.decode())));
  await lab.screenshot({ path: path.join(output, 'lab-import.png'), fullPage: true });
  await capturePage.reload();
  await until(async () => await capturePage.locator('#capture-results img').count() === 2);
  await capturePage.setViewportSize({ width: 390, height: 844 });
  assert(await capturePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile extension must not overflow horizontally');
  await capturePage.screenshot({ path: path.join(output, 'extension-mobile.png'), fullPage: true });
  await lab.setViewportSize({ width: 390, height: 844 });
  assert(await lab.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile capture import view must not overflow horizontally');
  await lab.screenshot({ path: path.join(output, 'lab-import-mobile.png'), fullPage: true });
  capturePage.once('dialog', dialog => dialog.dismiss());
  await capturePage.getByRole('button', { name: 'Delete batch', exact: true }).click();
  assert.equal(await capturePage.locator('#capture-results img').count(), 2, 'Dismissing delete must preserve screenshots');
  capturePage.once('dialog', dialog => dialog.accept());
  await capturePage.getByRole('button', { name: 'Delete batch', exact: true }).click();
  await until(async () => await capturePage.locator('#capture-results img').count() === 0 && await exportButton.isDisabled());
  assert.deepEqual(browserErrors, []);
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, real_debugger: true,
    extension_id: extensionId, viewports: inspected.images, original, restored, images: images.length,
    group: 'capture-fixture', reload_persistence: true, deletion_confirmed: true,
    mobile_overflow: false, browser_errors: browserErrors }, null, 2));
  console.log(`PASS: installed extension captured both CSS breakpoints, restored viewport/scroll, released debugger, exported/imported grouped images, persisted after reload, and passed mobile/deletion checks. Evidence: ${output}`);
} catch (error) {
  if (capturePage) await capturePage.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
  throw error;
} finally {
  if (context) await context.close();
  await new Promise(resolve => fixture.close(resolve));
  server.kill();
  await writeFile(path.join(output, 'server.log'), log);
}
