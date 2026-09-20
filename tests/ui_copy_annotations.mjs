// Isolated local UI regression: copy saved annotations, adapt them to another viewport, and save explicitly.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { spawn, spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'test-results', `copy-annotations-${Date.now()}`);
await mkdir(output, { recursive: true });
const python = process.platform === 'win32' ? path.join(repo, '.venv', 'Scripts', 'python.exe') : path.join(repo, '.venv', 'bin', 'python');
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const base = `http://127.0.0.1:${port}`;
const server = spawn(python, ['-m', 'uvicorn', 'grounding.api:app', '--host', '127.0.0.1', '--port', String(port)], {
  cwd: repo, env: { ...process.env, GROUNDING_DATA_DIR: path.join(output, 'data') }, windowsHide: true,
});
let log = '', browser, page;
const errors = [];
server.stdout.on('data', value => { log += value; });
server.stderr.on('data', value => { log += value; });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await fn(); if (value) return value; await pause(100); }
  throw new Error('Timed out waiting for annotation copy state');
}
async function api(route, payload, method = payload === undefined ? 'GET' : 'POST') {
  const response = await fetch(`${base}/api${route}`, { method, headers: { 'Content-Type': 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload) });
  assert(response.ok, `${method} ${route}: ${response.status} ${await response.clone().text()}`);
  return response.json();
}
async function upload(name, width, height, color) {
  const generated = spawnSync(python, ['-c', `import io,sys
from PIL import Image,ImageDraw
im=Image.new('RGB',(int(sys.argv[1]),int(sys.argv[2])),sys.argv[3])
draw=ImageDraw.Draw(im)
w,h=im.size
draw.rectangle((.2*w,.2*h,.5*w,.4*h),fill='#d5e9db',outline='#23664b',width=3)
draw.rectangle((.55*w,.5*h,.85*w,.7*h),fill='#f3e5c2',outline='#9c6a21',width=3)
im.save(sys.stdout.buffer,'PNG')`, String(width), String(height), color], { cwd: repo, windowsHide: true });
  assert.equal(generated.status, 0, generated.stderr?.toString());
  const form = new FormData();
  form.append('files', new Blob([generated.stdout], { type: 'image/png' }), name);
  const response = await fetch(`${base}/api/images`, { method: 'POST', body: form });
  assert(response.ok, await response.clone().text());
  return (await response.json())[0];
}
const sourceElements = [
  { id: 'source-action', class_id: 0, label: 'Primary action', bbox: [.2, .2, .5, .4], click_point: [.45, .35] },
  { id: 'source-search', class_id: 1, label: 'Search input', bbox: [.55, .5, .85, .7], click_point: [.65, .6] },
];
const sourceExamples = [
  { id: 'source-primary', instruction: 'Click the primary action', target_present: true, element_id: 'source-action', status: 'reviewed', ambiguous: false },
  { id: 'source-ambiguous', instruction: 'Maybe use this primary action', target_present: true, element_id: 'source-action', status: 'draft', ambiguous: true },
  { id: 'source-search-example', instruction: 'Find the search input', target_present: true, element_id: 'source-search', status: 'reviewed', ambiguous: false },
  { id: 'source-absent', instruction: 'Find the missing toggle', target_present: false, element_id: null, status: 'reviewed', ambiguous: false },
];
function assertNear(actual, expected, message) { assert(Math.abs(actual - expected) < .007, `${message}: ${actual} != ${expected}`); }
try {
  await until(async () => { try { return (await fetch(`${base}/api/health`)).ok; } catch { return false; } });
  const sourceUpload = await upload('source-wide.png', 800, 500, '#eef3ef');
  const source = await api(`/images/${sourceUpload.id}`, { elements: sourceElements, examples: sourceExamples, group: 'source-page' }, 'PUT');
  const destinationUpload = await upload('destination-tall.png', 400, 800, '#f7f5ef');
  const existingElement = { id: 'destination-existing', class_id: 2, label: 'Existing destination target', bbox: [.65, .76, .9, .88], click_point: [.8, .82] };
  const existingExample = { id: 'destination-existing-example', instruction: 'Use the existing destination target', target_present: true,
    element_id: existingElement.id, status: 'reviewed', ambiguous: false };
  const destination = await api(`/images/${destinationUpload.id}`, { elements: [existingElement], examples: [existingExample], group: 'destination-page' }, 'PUT');
  const partialDestination = await upload('partial-destination.png', 600, 420, '#f3f0f8');

  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(base);
  await page.locator('nav').getByRole('button', { name: /^Images & annotations/ }).click();
  await page.getByRole('button', { name: 'Open destination-tall.png', exact: true }).click();
  const canvas = page.getByRole('img', { name: 'Interactive screenshot annotation canvas' });
  await canvas.waitFor();
  const groupField = page.getByLabel('Website / template / session group', { exact: false });
  await groupField.fill('destination-unsaved-group');

  async function openCopy(sourceId = source.id) {
    await page.getByRole('button', { name: 'Copy from image', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Copy annotations from another image', exact: true });
    await dialog.waitFor();
    const sourceSelect = dialog.getByLabel('Source image', { exact: true });
    if (await sourceSelect.inputValue() !== sourceId) await sourceSelect.selectOption(sourceId);
    await dialog.getByRole('checkbox', { name: 'Copy element 1: Primary action', exact: true }).waitFor();
    return dialog;
  }
  let dialog = await openCopy();
  const sourceReload = page.waitForResponse(response => response.url().endsWith(`/api/images/${source.id}`) && response.request().method() === 'GET');
  await dialog.getByLabel('Source image', { exact: true }).selectOption(source.id);
  assert((await sourceReload).ok());
  await dialog.getByRole('checkbox', { name: 'Copy element 1: Primary action', exact: true }).waitFor();
  assert(await dialog.getByRole('checkbox', { name: 'Copy element 1: Primary action', exact: true }).isChecked());
  assert(await dialog.getByRole('checkbox', { name: 'Copy element 2: Search input', exact: true }).isChecked());
  assert(await dialog.getByRole('checkbox', { name: 'Include absent-target instructions', exact: true }).isChecked());
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.equal(await page.locator('.element-list .element-item').count(), 1);
  assert.equal(await groupField.inputValue(), 'destination-unsaved-group');
  assert.deepEqual(await api(`/images/${destination.id}`), destination);

  dialog = await openCopy();
  await dialog.getByRole('button', { name: 'Add to this image', exact: true }).click();
  await until(async () => await page.locator('.element-list .element-item').count() === 3);
  assert.equal(await groupField.inputValue(), 'destination-unsaved-group', 'Copy must preserve unsaved destination edits');
  assert.deepEqual(await api(`/images/${destination.id}`), destination, 'Copy must not persist before Save');
  assert.deepEqual(await api(`/images/${source.id}`), source, 'Copy must never edit the source');
  for (const [index, label] of ['X min', 'Y min', 'X max', 'Y max'].entries()) {
    assertNear(Number(await page.getByLabel(label, { exact: true }).inputValue()), sourceElements[0].bbox[index], label);
  }
  await page.getByText('Pixels: 80, 160, 200, 320', { exact: true }).waitFor();
  await page.getByRole('button', { name: /^2\. Instructions/ }).click();
  assert.equal(await page.locator('.instruction-card').count(), 5);
  await page.getByRole('button', { name: /^1\. Elements/ }).click();
  await page.locator('.element-list .element-item').filter({ hasText: 'Primary action' }).click();
  await page.getByTitle('Zoom in', { exact: true }).click();
  await page.getByTitle('Zoom in', { exact: true }).click();
  async function screenPoint(x, y) {
    return canvas.evaluate((svg, position) => {
      const point = new DOMPoint(position[0] * 400, position[1] * 800).matrixTransform(svg.querySelector('g').getScreenCTM());
      return [point.x, point.y];
    }, [x, y]);
  }
  await page.mouse.move(...await screenPoint(.5, .4));
  await page.mouse.down();
  await page.mouse.move(...await screenPoint(.4, .3), { steps: 8 });
  await page.mouse.up();
  assertNear(Number(await page.getByLabel('X max', { exact: true }).inputValue()), .4, 'Zoomed resized X max');
  assertNear(Number(await page.getByLabel('Y max', { exact: true }).inputValue()), .3, 'Zoomed resized Y max');
  // Shrinking clamps the click point onto this corner; the corner must remain draggable above it.
  await page.mouse.move(...await screenPoint(.4, .3));
  await page.mouse.down();
  await page.mouse.move(...await screenPoint(.46, .34), { steps: 8 });
  await page.mouse.up();
  assertNear(Number(await page.getByLabel('X max', { exact: true }).inputValue()), .46, 'Repeated corner resize X max');
  assertNear(Number(await page.getByLabel('Y max', { exact: true }).inputValue()), .34, 'Repeated corner resize Y max');
  await page.getByLabel('X max', { exact: true }).fill('0.25');
  await page.getByLabel('Y max', { exact: true }).fill('0.24');
  assert(Number(await page.getByLabel('X', { exact: true }).inputValue()) <= .25, 'Numeric box shrink must keep the click X inside');
  assert(Number(await page.getByLabel('Y', { exact: true }).inputValue()) <= .24, 'Numeric box shrink must keep the click Y inside');
  await page.getByLabel('X max', { exact: true }).fill('0.4');
  await page.getByLabel('Y max', { exact: true }).fill('0.3');
  let releaseSave;
  const saveGate = new Promise(resolve => { releaseSave = resolve; });
  const destinationRoute = `${base}/api/images/${destination.id}`;
  await page.route(destinationRoute, async route => {
    if (route.request().method() === 'PUT') await saveGate;
    await route.continue();
  });
  const saveResponse = page.waitForResponse(response => response.url().endsWith(`/api/images/${destination.id}`) && response.request().method() === 'PUT');
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const copyingDisabledDuringSave = await page.getByRole('button', { name: 'Copy from image', exact: true }).isDisabled();
  releaseSave();
  assert(copyingDisabledDuringSave, 'Copy must wait for the pending Save');
  assert((await saveResponse).ok());
  await page.unroute(destinationRoute);
  const saved = await api(`/images/${destination.id}`);
  assert.equal(saved.elements.length, 3);
  assert.equal(saved.examples.length, 5);
  assert.equal(saved.group, 'destination-unsaved-group');
  assert.deepEqual(saved.elements[0], existingElement);
  assert.deepEqual(saved.examples[0], existingExample);
  const copiedElements = saved.elements.slice(1);
  assert(copiedElements.every(element => !sourceElements.some(original => original.id === element.id) && element.id !== existingElement.id));
  assert.equal(new Set(saved.elements.map(element => element.id)).size, 3);
  for (const original of sourceExamples) {
    const copied = saved.examples.find(example => example.instruction === original.instruction);
    assert(copied && copied.id !== original.id);
    assert.equal(copied.status, 'draft');
    assert.equal(copied.ambiguous, original.ambiguous);
    assert.equal(copied.target_present, original.target_present);
    const oldTarget = sourceElements.find(element => element.id === original.element_id);
    assert.equal(copied.element_id, oldTarget ? copiedElements.find(element => element.label === oldTarget.label).id : null);
  }
  assert.equal(new Set(saved.examples.map(example => example.id)).size, 5);
  const resized = copiedElements.find(element => element.label === 'Primary action');
  [.2, .2, .4, .3].forEach((value, index) => assertNear(resized.bbox[index], value, `Resized coordinate ${index}`));
  assert(resized.click_point[0] >= resized.bbox[0] && resized.click_point[0] <= resized.bbox[2]);
  assert(resized.click_point[1] >= resized.bbox[1] && resized.click_point[1] <= resized.bbox[3]);
  assert.deepEqual(copiedElements.find(element => element.label === 'Search input').bbox, sourceElements[1].bbox);
  assert.deepEqual(await api(`/images/${source.id}`), source);
  await page.screenshot({ path: path.join(output, 'copied-resized.png'), fullPage: true });

  await page.reload();
  await page.getByRole('button', { name: 'Open destination-tall.png', exact: true }).click();
  await canvas.waitFor();
  assert.equal(await page.locator('.element-list .element-item').count(), 3);
  assert.deepEqual(await api(`/images/${destination.id}`), saved);
  await page.getByTitle('Close workspace', { exact: true }).click();
  await page.getByRole('button', { name: 'Open partial-destination.png', exact: true }).click();
  dialog = await openCopy();
  await dialog.getByRole('checkbox', { name: 'Copy element 1: Primary action', exact: true }).uncheck();
  await dialog.getByRole('checkbox', { name: 'Include absent-target instructions', exact: true }).uncheck();
  await page.setViewportSize({ width: 390, height: 844 });
  const bounds = await dialog.boundingBox();
  assert(bounds && bounds.x >= 0 && bounds.x + bounds.width <= 391, 'Copy dialog must fit a narrow viewport');
  assert(await dialog.getByRole('button', { name: 'Add to this image', exact: true }).isVisible());
  await page.screenshot({ path: path.join(output, 'copy-modal-mobile.png'), fullPage: true });
  await dialog.getByRole('button', { name: 'Add to this image', exact: true }).click();
  assert.deepEqual(await api(`/images/${partialDestination.id}`), partialDestination);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const partialSave = page.waitForResponse(response => response.url().endsWith(`/api/images/${partialDestination.id}`) && response.request().method() === 'PUT');
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  assert((await partialSave).ok());
  const partial = await api(`/images/${partialDestination.id}`);
  assert.equal(partial.elements.length, 1);
  assert.equal(partial.examples.length, 1);
  assert.equal(partial.elements[0].label, 'Search input');
  assert.equal(partial.examples[0].instruction, 'Find the search input');
  assert.equal(partial.examples[0].element_id, partial.elements[0].id);
  assert.equal(partial.examples[0].status, 'draft');
  assert.equal(partial.group, partialDestination.group);
  assert.deepEqual(await api(`/images/${source.id}`), source);
  assert.deepEqual(errors, []);
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, source_id: source.id,
    destination_id: destination.id, source_dimensions: [source.width, source.height],
    destination_dimensions: [destination.width, destination.height], copied_elements: 2, copied_instructions: 4,
    saved_resized_box: resized.bbox, source_unchanged: true, explicit_save: true, partial_copy: true,
    zoomed_repeated_resize: true, numeric_click_clamp: true, mobile_dialog_fits: true,
    same_source_reselect: true, browser_errors: errors }, null, 2));
  console.log(`PASS: copied saved annotations across dimensions, preserved unsaved edits/source, resized under zoom, saved/reloaded drafts, and verified partial selection/mobile dialog. Evidence: ${output}`);
} catch (error) {
  if (page) await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill();
  await writeFile(path.join(output, 'server.log'), log);
}
