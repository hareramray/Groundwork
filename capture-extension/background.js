import {runCapture, validateSettings} from "./capture-core.js";
import * as store from "./store.js";

let active = null;
let launchPending = false;
const extensionRoot = chrome.runtime.getURL("");

function notify(batch) {
  return chrome.runtime.sendMessage({type: "capture:updated", batchId: batch.id}).catch(() => {});
}

// An interrupted browser/worker never leaves a batch looking permanently active.
const recovered = (async () => {
  for (const batch of await store.listBatches()) {
    if (batch.status !== "capturing") continue;
    const target = {tabId: batch.tab_id};
    // Commands can only use a debugger attached by this extension. If the browser
    // already detached it, these harmless cleanup attempts simply reject.
    await chrome.debugger.sendCommand(target, "Emulation.clearDeviceMetricsOverride", {}).catch(() => {});
    const scroll = batch.original_scroll;
    if (scroll && Number.isFinite(scroll.x) && Number.isFinite(scroll.y)) {
      await chrome.debugger.sendCommand(target, "Runtime.evaluate", {
        expression: `window.scrollTo({left:${scroll.x},top:${scroll.y},behavior:'instant'});`, returnByValue: true,
      }).catch(() => {});
    }
    await chrome.debugger.detach(target).catch(() => {});
    batch.status = "error";
    batch.completed = batch.captures?.length || 0;
    batch.message = "Capture was interrupted. Completed screenshots are saved; start a new batch to continue.";
    batch.errors = [...(batch.errors || []), batch.message];
    await store.saveBatch(batch);
  }
})();

function website(url) {
  try { return ["http:", "https:"].includes(new URL(url).protocol); }
  catch { return false; }
}

async function start(input) {
  if (active || launchPending) throw new Error("A capture is already running. Finish or cancel it before starting another.");
  launchPending = true;
  try {
    await recovered;
    const settings = validateSettings(input);
    const tab = await chrome.tabs.get(settings.tabId);
    if (!website(tab.url)) throw new Error("Select a normal HTTP or HTTPS website tab. Browser settings and extension pages cannot be captured.");
    if (tab.url.length > 8192) throw new Error("This page URL is too long to include in a training capture bundle.");
    const url = new URL(tab.url);
    settings.group ||= (url.origin + url.pathname).slice(0, 500);
    const batch = {id: crypto.randomUUID(), name: settings.name, group: settings.group,
      created_at: new Date().toISOString(), status: "capturing", captures: [], errors: [],
      tab_id: tab.id, source_url: tab.url, title: (tab.title || "").slice(0, 1000), resolutions: settings.resolutions,
      completed: 0, total: settings.resolutions.length, message: "Connecting to the selected website…"};
    await store.saveBatch(batch);
    const job = {batchId: batch.id, tabId: tab.id, controller: new AbortController(), detaching: false};
    active = job;
    const adapter = {
      async attach(tabId) {
        try { await chrome.debugger.attach({tabId}, "1.3"); }
        catch (error) { throw new Error("Cannot capture this tab. Close its DevTools or another debugger and try again. " + error.message); }
      },
      send: (tabId, method, params) => chrome.debugger.sendCommand({tabId}, method, params),
      async detach(tabId) { job.detaching = true; await chrome.debugger.detach({tabId}); },
      sleep: milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds)),
    };
    void runCapture({settings, batch, adapter, store, signal: job.controller.signal, onUpdate: notify})
      .catch(async error => {
        batch.status = "error";
        batch.message = "Could not save the capture: " + (error.message || String(error));
        batch.errors.push(batch.message);
        await store.saveBatch(batch).catch(() => {});
        await notify(batch);
      }).finally(() => { if (active === job) active = null; });
    return {ok: true, batchId: batch.id};
  } finally { launchPending = false; }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (!sender || sender.id !== chrome.runtime.id || typeof sender.url !== "string" || !sender.url.startsWith(extensionRoot)) return false;
  if (!["capture:start", "capture:cancel", "capture:status"].includes(message?.type)) return false;
  void (async () => {
    if (message.type === "capture:start") return start(message);
    await recovered;
    if (message.type === "capture:cancel") {
      if (!active || active.batchId !== message.batchId) throw new Error("That batch is no longer capturing.");
      active.controller.abort(new DOMException("Capture canceled.", "AbortError"));
    }
    return {ok: true, activeBatchId: active?.batchId || null};
  })().then(respond, error => respond({ok: false, error: error.message || String(error)}));
  return true;
});

function abortTab(tabId, message) {
  if (active?.tabId === tabId && !active.detaching) active.controller.abort(new Error(message));
}

chrome.debugger.onDetach.addListener((source, reason) => {
  abortTab(source.tabId, `Capture stopped because the debugger detached (${reason}). Completed screenshots are saved.`);
});
chrome.debugger.onEvent.addListener((source, method, params) => {
  if (method === "Page.frameNavigated" && params.frame && !params.frame.parentId) {
    abortTab(source.tabId, "Capture stopped because the selected website navigated. Completed screenshots are saved.");
  }
});
chrome.tabs.onRemoved.addListener(tabId => abortTab(tabId, "The selected tab was closed. Completed screenshots are saved."));
chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (change.url || change.status === "loading") abortTab(tabId, "Capture stopped because the selected website navigated. Completed screenshots are saved.");
});
chrome.action.onClicked.addListener(tab => {
  const query = Number.isSafeInteger(tab?.id) && website(tab.url) ? `?tabId=${tab.id}` : "";
  void chrome.tabs.create({url: chrome.runtime.getURL("capture.html") + query});
});
