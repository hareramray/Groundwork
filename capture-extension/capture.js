import { deleteBatch, getBatch, getImage, listBatches } from './store.js';
import { buildZip } from './zip.js';

const byId = id => document.getElementById(id);
const elements = {
  form: byId('capture-form'), settings: byId('capture-settings'), source: byId('source-tab'),
  refresh: byId('refresh-tabs'), sourceUrl: byId('source-url'), name: byId('capture-name'),
  group: byId('dataset-group'), delay: byId('settle-delay'), presets: byId('presets'),
  custom: byId('custom-sizes'), width: byId('custom-width'), height: byId('custom-height'),
  add: byId('add-size'), summary: byId('size-summary'), start: byId('capture-start'),
  cancel: byId('capture-cancel'), history: byId('batch-history'), notice: byId('notice'),
  batchName: byId('batch-name'), batchStatus: byId('batch-status'), description: byId('batch-description'),
  progress: byId('capture-progress'), progressText: byId('capture-progress-text'),
  errors: byId('batch-errors'), export: byId('export-zip'), results: byId('capture-results'),
  resultCount: byId('results-count'),
  delete: byId('delete-batch'),
};
const presetInputs = [...document.querySelectorAll('input[name="preset"]')];
const imageCache = new Map();
let tabs = [];
let customSizes = [];
let batches = [];
let selectedBatchId = '';
let activeBatchId = null;
let starting = false;
let canceling = false;
let exporting = false;
let deleting = false;
let refreshing = false;
let rerenderRequested = false;
let initialized = false;
let lastResultSignature = '';
let autoName = true;
let autoGroup = true;
let renderRevision = 0;

function message(error) { return error instanceof Error ? error.message : String(error); }
function showError(error) {
  elements.notice.textContent = message(error);
  elements.notice.hidden = false;
}
function clearError() { elements.notice.hidden = true; elements.notice.textContent = ''; }
function sizeFromValue(value) {
  const [width, height] = value.split('x').map(Number);
  return { width, height };
}
function selectedSizes() {
  const sizes = [...presetInputs.filter(input => input.checked).map(input => sizeFromValue(input.value)), ...customSizes];
  return [...new Map(sizes.map(size => [`${size.width}x${size.height}`, size])).values()];
}
function checkSizes(sizes) {
  if (!sizes.length) throw new Error('Select at least one viewport size.');
  if (sizes.length > 8) throw new Error('Select no more than 8 viewport sizes in one batch.');
  if (sizes.some(size => !Number.isInteger(size.width) || !Number.isInteger(size.height) || size.width < 240 || size.height < 240 || size.width > 4096 || size.height > 4096)) {
    throw new Error('Each width and height must be a whole number from 240 to 4096 pixels.');
  }
  if (sizes.reduce((total, size) => total + size.width * size.height, 0) > 40_000_000) {
    throw new Error('This batch exceeds 40 million pixels. Remove a size or choose smaller dimensions.');
  }
}
function updateControls() {
  const capturing = Boolean(activeBatchId) || starting;
  elements.settings.disabled = !initialized || capturing;
  elements.history.disabled = !batches.length || capturing;
  elements.start.disabled = !tabs.length || !selectedSizes().length || capturing || deleting;
  elements.start.textContent = starting ? 'Starting capture…' : capturing ? 'Capture in progress…' : 'Capture selected sizes';
  elements.cancel.hidden = !activeBatchId;
  elements.cancel.disabled = canceling;
  elements.cancel.textContent = canceling ? 'Canceling…' : 'Cancel capture';
  const selected = batches.find(batch => batch.id === selectedBatchId);
  elements.export.disabled = !selected?.captures?.length || exporting || deleting;
  elements.export.textContent = exporting ? 'Preparing ZIP…' : 'Export capture ZIP';
  elements.delete.disabled = !selected || selected.status === 'capturing' || capturing || exporting || deleting;
  elements.delete.textContent = deleting ? 'Deleting…' : 'Delete batch';
}
function updateSizeSummary() {
  const sizes = selectedSizes();
  const pixels = sizes.reduce((total, size) => total + size.width * size.height, 0);
  elements.summary.textContent = `${sizes.length} selected · ${(pixels / 1_000_000).toFixed(1)} MP`;
  updateControls();
}
function renderCustomSizes() {
  elements.custom.replaceChildren();
  for (const size of customSizes) {
    const chip = document.createElement('span');
    chip.className = 'custom-chip';
    chip.append(document.createTextNode(`${size.width} × ${size.height}`));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove ${size.width} by ${size.height} size`);
    remove.addEventListener('click', () => {
      customSizes = customSizes.filter(item => item !== size);
      renderCustomSizes(); updateSizeSummary();
    });
    chip.append(remove); elements.custom.append(chip);
  }
}
function applyTab() {
  const tab = tabs.find(item => String(item.id) === elements.source.value);
  if (!tab) { elements.sourceUrl.textContent = 'Open a normal webpage, then refresh this list.'; return; }
  elements.sourceUrl.textContent = tab.url;
  if (autoName || !elements.name.value.trim()) elements.name.value = (tab.title || new URL(tab.url).hostname).slice(0, 160);
  if (autoGroup || !elements.group.value.trim()) {
    const url = new URL(tab.url);
    elements.group.value = `${url.origin}${url.pathname}`.slice(0, 500);
  }
}
async function refreshTabs() {
  const previous = elements.source.value;
  const requested = new URL(location.href).searchParams.get('tabId');
  tabs = (await chrome.tabs.query({})).filter(tab => Number.isInteger(tab.id) && /^https?:\/\//i.test(tab.url || ''));
  elements.source.replaceChildren();
  if (!tabs.length) {
    const option = document.createElement('option'); option.value = ''; option.textContent = 'No open webpages found';
    elements.source.append(option);
  } else {
    for (const tab of tabs) {
      const option = document.createElement('option'); option.value = String(tab.id);
      option.textContent = `${tab.title || 'Untitled page'} · ${new URL(tab.url).hostname}`;
      elements.source.append(option);
    }
    elements.source.value = [previous, requested, String(tabs.find(tab => tab.active)?.id || ''), String(tabs[0].id)].find(id => tabs.some(tab => String(tab.id) === id));
  }
  applyTab(); updateControls();
}
function dateLabel(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
async function readImage(id) {
  if (!imageCache.has(id)) {
    const data = await getImage(id);
    if (typeof data !== 'string' || !data.startsWith('data:image/png;base64,')) throw new Error('A saved PNG could not be loaded.');
    imageCache.set(id, data);
  }
  return imageCache.get(id);
}
async function renderResults(batch) {
  const captures = batch?.captures || [];
  const signature = `${batch?.id || ''}:${captures.map(capture => capture.id).join(',')}`;
  elements.resultCount.textContent = `${captures.length} ${captures.length === 1 ? 'image' : 'images'}`;
  if (signature === lastResultSignature) return;
  lastResultSignature = signature;
  const revision = ++renderRevision;
  if (!captures.length) {
    const empty = document.createElement('div'); empty.className = 'empty-results';
    const icon = document.createElement('span'); icon.className = 'empty-screen'; icon.setAttribute('aria-hidden', 'true');
    const title = document.createElement('h3'); title.textContent = 'A view for every screen';
    const detail = document.createElement('p'); detail.textContent = 'Your screenshots will appear here as each capture completes.';
    empty.append(icon, title, detail); elements.results.replaceChildren(empty); return;
  }
  const cards = await Promise.all(captures.map(async capture => {
    const card = document.createElement('article'); card.className = 'capture-result';
    const preview = document.createElement('div'); preview.className = 'preview';
    try {
      const image = document.createElement('img');
      image.src = await readImage(capture.id);
      image.alt = `${capture.title || 'Webpage'} captured at ${capture.width} by ${capture.height} pixels`;
      preview.append(image);
    } catch (error) { const note = document.createElement('p'); note.className = 'small muted'; note.textContent = message(error); preview.append(note); }
    const detail = document.createElement('div'); detail.className = 'result-detail';
    const row = document.createElement('div');
    const title = document.createElement('h3'); title.textContent = `${capture.width} × ${capture.height}`;
    const format = document.createElement('small'); format.textContent = 'PNG';
    row.append(title, format);
    const viewport = document.createElement('p'); viewport.textContent = `Viewport ${capture.viewport_width} × ${capture.viewport_height} · ${dateLabel(capture.captured_at)}`;
    detail.append(row, viewport); card.append(preview, detail); return card;
  }));
  if (revision === renderRevision) elements.results.replaceChildren(...cards);
}
async function renderBatch() {
  const batch = batches.find(item => item.id === selectedBatchId);
  elements.history.replaceChildren();
  if (!batches.length) {
    const empty = document.createElement('option'); empty.value = ''; empty.textContent = 'No captures yet'; elements.history.append(empty);
  } else {
    for (const item of batches) {
      const option = document.createElement('option'); option.value = item.id;
      option.textContent = `${item.name} · ${dateLabel(item.created_at)}`; elements.history.append(option);
    }
    elements.history.value = selectedBatchId;
  }
  elements.batchName.textContent = batch?.name || 'Ready when you are';
  const status = batch?.status || 'ready';
  elements.batchStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  elements.batchStatus.dataset.status = status;
  elements.description.textContent = batch ? `Group: ${batch.group}` : 'Your screenshots are saved locally as each size completes.';
  const completed = batch?.completed || 0;
  const total = batch?.total || batch?.resolutions?.length || 0;
  elements.progress.max = Math.max(1, total);
  elements.progress.value = Math.min(completed, total);
  elements.progressText.textContent = batch ? `${completed} / ${total} sizes processed · ${batch.message || `${batch.captures?.length || 0} screenshots saved`}` : 'Choose your page and capture sizes to begin.';
  elements.errors.replaceChildren();
  for (const error of batch?.errors || []) {
    const text = document.createElement('p');
    text.textContent = typeof error === 'string' ? error : (error.message || error.error || JSON.stringify(error));
    elements.errors.append(text);
  }
  elements.errors.hidden = !batch?.errors?.length;
  updateControls();
  await renderResults(batch);
}
async function refreshState() {
  if (refreshing) { rerenderRequested = true; return; }
  refreshing = true;
  try {
    const [status, saved] = await Promise.all([chrome.runtime.sendMessage({ type: 'capture:status' }), listBatches()]);
    if (!status?.ok) throw new Error(status?.error || 'The capture worker could not be reached.');
    activeBatchId = status.activeBatchId || null;
    batches = saved.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    if (activeBatchId) selectedBatchId = activeBatchId;
    else if (!batches.some(batch => batch.id === selectedBatchId)) selectedBatchId = batches[0]?.id || '';
    await renderBatch();
  } catch (error) { showError(error); }
  finally {
    refreshing = false;
    if (rerenderRequested) { rerenderRequested = false; void refreshState(); }
  }
}
async function startCapture(event) {
  event.preventDefault();
  if (starting || activeBatchId) return;
  clearError();
  try {
    const resolutions = selectedSizes(); checkSizes(resolutions);
    const tabId = Number(elements.source.value);
    if (!tabs.some(tab => tab.id === tabId)) throw new Error('Choose an open webpage to capture.');
    const name = elements.name.value.trim(); const group = elements.group.value.trim();
    if (!name || !group) throw new Error('Enter a capture name and dataset group.');
    const delayMs = Number(elements.delay.value);
    if (!Number.isInteger(delayMs) || delayMs < 250 || delayMs > 10000) throw new Error('Settle delay must be a whole number from 250 to 10000 milliseconds.');
    starting = true; updateControls();
    const result = await chrome.runtime.sendMessage({ type: 'capture:start', tabId, name, group, resolutions, delayMs });
    if (!result?.ok) throw new Error(result?.error || 'Capture could not be started.');
    activeBatchId = result.batchId; selectedBatchId = result.batchId;
    await refreshState();
  } catch (error) { showError(error); }
  finally { starting = false; updateControls(); }
}
async function cancelCapture() {
  if (!activeBatchId || canceling) return;
  canceling = true; clearError(); updateControls();
  try {
    const result = await chrome.runtime.sendMessage({ type: 'capture:cancel', batchId: activeBatchId });
    if (!result?.ok) throw new Error(result?.error || 'Capture could not be canceled.');
    await refreshState();
  } catch (error) { showError(error); }
  finally { canceling = false; updateControls(); }
}
async function exportBatch() {
  if (!selectedBatchId || exporting) return;
  exporting = true; clearError(); updateControls();
  try {
    const batch = await getBatch(selectedBatchId);
    if (!batch?.captures?.length) throw new Error('This batch has no saved screenshots yet.');
    const manifest = { format: 'groundwork-captures', schema_version: 1, id: batch.id, name: batch.name, group: batch.group, created_at: batch.created_at, captures: batch.captures };
    const entries = [{ name: 'manifest.json', data: JSON.stringify(manifest, null, 2) }];
    for (const capture of batch.captures) {
      const dataUrl = await readImage(capture.id);
      const decoded = atob(dataUrl.slice(dataUrl.indexOf(',') + 1));
      const bytes = Uint8Array.from(decoded, character => character.charCodeAt(0));
      entries.push({ name: capture.image_path, data: bytes });
    }
    const blob = await buildZip(entries);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url;
    const filename = batch.name.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'groundwork-captures';
    link.download = `${filename}-${batch.id.slice(0, 8)}.zip`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (error) { showError(error); }
  finally { exporting = false; updateControls(); }
}
async function removeBatch() {
  const batch = batches.find(item => item.id === selectedBatchId);
  if (!batch || activeBatchId || starting || exporting || deleting || batch.status === 'capturing') return;
  if (!confirm(`Delete "${batch.name}" and its saved screenshots from this browser? Exported ZIP files are kept.`)) return;
  deleting = true; clearError(); updateControls();
  try {
    await deleteBatch(batch.id);
    for (const capture of batch.captures || []) imageCache.delete(capture.id);
    selectedBatchId = '';
    await refreshState();
  } catch (error) { showError(error); }
  finally { deleting = false; updateControls(); }
}

for (const input of presetInputs) {
  const size = sizeFromValue(input.value); input.setAttribute('aria-label', `${size.width} × ${size.height}`);
  input.addEventListener('change', updateSizeSummary);
}
elements.form.addEventListener('submit', event => void startCapture(event));
elements.refresh.addEventListener('click', () => { clearError(); void refreshTabs().catch(showError); });
elements.source.addEventListener('change', () => { autoName = true; autoGroup = true; applyTab(); });
elements.name.addEventListener('input', () => { autoName = false; });
elements.group.addEventListener('input', () => { autoGroup = false; });
elements.add.addEventListener('click', () => {
  clearError();
  try {
    const size = { width: Number(elements.width.value), height: Number(elements.height.value) };
    checkSizes([size]);
    if (selectedSizes().some(item => item.width === size.width && item.height === size.height)) throw new Error('That viewport size is already selected.');
    checkSizes([...selectedSizes(), size]);
    const preset = presetInputs.find(input => input.value === `${size.width}x${size.height}`);
    if (preset) preset.checked = true; else customSizes.push(size);
    elements.width.value = ''; elements.height.value = '';
    renderCustomSizes(); updateSizeSummary();
  } catch (error) { showError(error); }
});
elements.history.addEventListener('change', () => { selectedBatchId = elements.history.value; void renderBatch().catch(showError); });
elements.cancel.addEventListener('click', () => void cancelCapture());
elements.export.addEventListener('click', () => void exportBatch());
elements.delete.addEventListener('click', () => void removeBatch());
chrome.runtime.onMessage.addListener(event => { if (event?.type === 'capture:updated') void refreshState(); });

try {
  await refreshTabs(); await refreshState();
  initialized = true; updateSizeSummary();
} catch (error) { showError(error); }
const poll = setInterval(() => void refreshState(), 1200);
window.addEventListener('pagehide', () => clearInterval(poll), { once: true });
