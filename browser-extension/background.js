"use strict";

// This worker accepts a fixed action vocabulary from an authenticated loopback
// relay. It never reads page text as instructions or executes page JavaScript.
let generation = 0;
let activeConfig = null;
let browserId = null;
let pollController = null;
let running = false;
const attached = new Set();
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

function validateConfig(config) {
  let url;
  try { url = new URL(config.url); } catch { throw new Error("Enter the bridge address printed by the CLI."); }
  if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || !url.port || url.username || url.password || url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("The bridge must be http://127.0.0.1:PORT, as printed by the CLI.");
  }
  if (typeof config.token !== "string" || config.token.length < 20 || config.token.length > 200) throw new Error("Paste the complete session token printed by the CLI.");
  return {...config, url: url.origin, name: String(config.name || "Browser").slice(0, 80), enabled: true};
}

async function setStatus(message) {
  await chrome.storage.local.set({connectionStatus: message});
}

async function request(config, path, data, signal) {
  const response = await fetch(config.url + path, {
    method: data === undefined ? "GET" : "POST",
    headers: {"Authorization": "Bearer " + config.token, ...(data === undefined ? {} : {"Content-Type": "application/json"})},
    ...(data === undefined ? {} : {body: JSON.stringify(data)}),
    signal: signal || AbortSignal.timeout(10000),
    cache: "no-store",
    redirect: "error",
    credentials: "omit",
  });
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || `Bridge returned ${response.status}.`);
    error.status = response.status;
    throw error;
  }
  return result;
}

async function detach(tabId) {
  attached.delete(tabId);
  try { await chrome.debugger.detach({tabId}); } catch { /* Already closed or detached. */ }
}

async function detachAll() {
  for (const tabId of [...attached]) await detach(tabId);
}

async function stopConnection(message = "Disconnected. Your tabs remain open.") {
  generation += 1;
  running = false;
  pollController?.abort();
  const previous = activeConfig;
  const previousId = browserId;
  activeConfig = null;
  browserId = null;
  const saved = await chrome.storage.local.get("config");
  if (saved.config) await chrome.storage.local.set({config: {...saved.config, enabled: false}});
  await detachAll();
  if (previous && previousId) {
    // Complete unregister before the popup can reconnect the same instance ID.
    await request(previous, "/v1/unregister", {browser_id: previousId}, AbortSignal.timeout(2000)).catch(() => {});
  }
  await setStatus(message);
}

async function startConnection(input) {
  const config = validateConfig(input);
  generation += 1;
  const ownGeneration = generation;
  running = false;
  pollController?.abort();
  await detachAll();
  const saved = await chrome.storage.local.get("instanceId");
  const instanceId = saved.instanceId || crypto.randomUUID();
  await chrome.storage.local.set({instanceId, config});
  activeConfig = config;
  const registered = await request(config, "/v1/register", {instance_id: instanceId, name: config.name});
  if (ownGeneration !== generation) return;
  browserId = registered.browser_id;
  running = true;
  await setStatus(`Connected as ${config.name}. Select a website in the CLI.`);
  void pollLoop(ownGeneration, config, instanceId);
}

async function pollLoop(ownGeneration, config, instanceId) {
  let backoff = 1000;
  while (ownGeneration === generation) {
    try {
      pollController = new AbortController();
      const timeout = setTimeout(() => pollController?.abort(), 25000);
      let response;
      try {
        response = await request(config, "/v1/poll?browser_id=" + encodeURIComponent(browserId), undefined, pollController.signal);
      } finally { clearTimeout(timeout); }
      if (ownGeneration !== generation) break;
      backoff = 1000;
      if (response.job) {
        const job = response.job;
        let result = null;
        let error = null;
        try {
          if (typeof job.expires_at !== "number" || Date.now() >= job.expires_at * 1000) throw new Error("Command expired before execution; no action was taken.");
          result = await execute(job.command, job.params || {});
        } catch (caught) { error = caught.message || String(caught); }
        if (ownGeneration !== generation) break;
        try {
          await request(config, "/v1/result", {browser_id: browserId, job_id: job.id, result, error});
        } catch (caught) {
          // Commands are never automatically replayed after an uncertain result.
          if (caught.status !== 404) throw caught;
        }
      }
    } catch (error) {
      if (ownGeneration !== generation) break;
      if ([401, 403].includes(error.status)) {
        await stopConnection("Token rejected. Paste the current CLI token and connect again.");
        return;
      }
      await detachAll();
      await setStatus("CLI bridge unavailable. Reconnecting… Start the CLI and update its token if needed.");
      await delay(backoff);
      backoff = Math.min(backoff * 2, 10000);
      if (ownGeneration !== generation) break;
      try {
        const registered = await request(config, "/v1/register", {instance_id: instanceId, name: config.name});
        if (ownGeneration !== generation) break;
        browserId = registered.browser_id;
        await setStatus(`Connected as ${config.name}. Select a website in the CLI.`);
      } catch (retryError) {
        if ([401, 403].includes(retryError.status)) {
          await stopConnection("Token rejected. Paste the current CLI token and connect again.");
          return;
        }
      }
    }
  }
  if (ownGeneration === generation) running = false;
}

function tabNumber(value) {
  if (!/^\d+$/.test(String(value))) throw new Error("Invalid tab ID. List tabs and select a website again.");
  const id = Number(value);
  if (!Number.isSafeInteger(id)) throw new Error("Invalid tab ID.");
  return id;
}

async function attach(tabId) {
  if (attached.has(tabId)) return;
  try {
    await chrome.debugger.attach({tabId}, "1.3");
    attached.add(tabId);
  } catch (error) {
    throw new Error("Cannot control this tab. Choose a normal website, close its DevTools or another debugger, and select it again. " + error.message);
  }
}

async function send(tabId, method, params = {}) {
  await attach(tabId);
  return chrome.debugger.sendCommand({tabId}, method, params);
}

async function describe(tabId) {
  const tab = await chrome.tabs.get(tabId);
  const metrics = await send(tabId, "Page.getLayoutMetrics");
  const viewport = metrics.cssLayoutViewport || metrics.layoutViewport;
  return {tab_id: String(tabId), title: tab.title || "", url: tab.url || "", viewport_width: viewport.clientWidth, viewport_height: viewport.clientHeight, scroll_x: viewport.pageX || 0, scroll_y: viewport.pageY || 0};
}

function keyDefinition(name) {
  const special = {
    Enter: ["Enter", "Enter", 13, "\r"], Tab: ["Tab", "Tab", 9], Escape: ["Escape", "Escape", 27],
    Backspace: ["Backspace", "Backspace", 8], Delete: ["Delete", "Delete", 46], Insert: ["Insert", "Insert", 45],
    ArrowLeft: ["ArrowLeft", "ArrowLeft", 37], ArrowUp: ["ArrowUp", "ArrowUp", 38], ArrowRight: ["ArrowRight", "ArrowRight", 39], ArrowDown: ["ArrowDown", "ArrowDown", 40],
    Home: ["Home", "Home", 36], End: ["End", "End", 35], PageUp: ["PageUp", "PageUp", 33], PageDown: ["PageDown", "PageDown", 34],
    Space: [" ", "Space", 32, " "], Control: ["Control", "ControlLeft", 17], Shift: ["Shift", "ShiftLeft", 16], Alt: ["Alt", "AltLeft", 18], Meta: ["Meta", "MetaLeft", 91],
  };
  if (special[name]) return special[name];
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(name)) return [name, name, 111 + Number(name.slice(1))];
  if (/^[a-zA-Z]$/.test(name)) return [name, "Key" + name.toUpperCase(), name.toUpperCase().charCodeAt(0), name];
  if (/^Key[A-Z]$/.test(name)) return [name.slice(3).toLowerCase(), name, name.charCodeAt(3), name.slice(3).toLowerCase()];
  if (/^[0-9]$/.test(name)) return [name, "Digit" + name, name.charCodeAt(0), name];
  const punctuation = {"-": ["Minus", 189], "=": ["Equal", 187], ",": ["Comma", 188], ".": ["Period", 190], "/": ["Slash", 191], ";": ["Semicolon", 186], "'": ["Quote", 222], "[": ["BracketLeft", 219], "]": ["BracketRight", 221], "\\": ["Backslash", 220], "`": ["Backquote", 192]};
  if (punctuation[name]) return [name, ...punctuation[name], name];
  if ([...name].length === 1 && !/[\u0000-\u001f\u007f]/.test(name)) return [name, "", 0, name];
  throw new Error("Unsupported key. Use Enter, Tab, arrows, Escape, letters, or shortcuts such as Control+A.");
}

async function press(tabId, shortcut) {
  if (typeof shortcut !== "string" || shortcut.length > 80) throw new Error("Invalid keyboard shortcut.");
  const parts = shortcut.split("+");
  const keyName = parts.pop();
  const modifierBits = {Alt: 1, Control: 2, Meta: 4, Shift: 8};
  let modifiers = 0;
  for (let part of parts) {
    if (part === "ControlOrMeta") part = navigator.platform.includes("Mac") ? "Meta" : "Control";
    if (!modifierBits[part]) throw new Error("Use canonical modifiers: Control, Alt, Shift, or Meta.");
    modifiers |= modifierBits[part];
  }
  let [key, code, windowsVirtualKeyCode, character] = keyDefinition(keyName);
  if (modifiers & 8 && character) {
    const shifted = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+", "[": "{", "]": "}", "\\": "|", ";": ":", "'": '"', ",": "<", ".": ">", "/": "?", "`": "~"};
    character = shifted[character] || character.toUpperCase();
    key = character;
  }
  const text = modifiers & (1 | 2 | 4) ? "" : character || "";
  const params = {key, code, windowsVirtualKeyCode, modifiers};
  await send(tabId, "Input.dispatchKeyEvent", {...params, type: text ? "keyDown" : "rawKeyDown", ...(text ? {text, unmodifiedText: character} : {})});
  await send(tabId, "Input.dispatchKeyEvent", {...params, type: "keyUp"});
}

async function navigate(tabId, value) {
  let url;
  try { url = new URL(value); } catch { throw new Error("Navigation requires an http:// or https:// URL."); }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) throw new Error("Navigation requires HTTP(S) without embedded credentials.");
  await chrome.tabs.update(tabId, {url: url.href});
  const deadline = Date.now() + 24000;
  while (Date.now() < deadline) {
    await delay(100);
    const tab = await chrome.tabs.get(tabId);
    if (tab.status === "complete" && !tab.pendingUrl) return;
  }
  throw new Error("Navigation timed out. Inspect the tab before retrying.");
}

async function execute(command, params) {
  if (command === "list_tabs") {
    const tabs = await chrome.tabs.query({});
    return tabs.map(tab => ({id: String(tab.id), title: tab.title || "", url: tab.url || tab.pendingUrl || "", active: Boolean(tab.active), window_id: tab.windowId}));
  }
  const tabId = tabNumber(params.tab_id);
  if (command === "detach") { await detach(tabId); return {}; }
  if (command === "select_tab") {
    await attach(tabId);
    const tab = await chrome.tabs.update(tabId, {active: true});
    await chrome.windows.update(tab.windowId, {focused: true});
    return describe(tabId);
  }
  if (command === "describe") return describe(tabId);
  if (command === "capture") {
    const info = await describe(tabId);
    const clip = {x: info.scroll_x, y: info.scroll_y, width: info.viewport_width, height: info.viewport_height, scale: 1};
    const screenshot = await send(tabId, "Page.captureScreenshot", {format: "png", fromSurface: true, captureBeyondViewport: false, clip});
    return {...info, data: screenshot.data};
  }
  if (command === "click") {
    const info = await describe(tabId);
    const {x, y} = params;
    if (![x, y].every(Number.isFinite) || x < 0 || y < 0 || x >= info.viewport_width || y >= info.viewport_height) throw new Error("Click is outside the visible viewport.");
    await send(tabId, "Input.dispatchMouseEvent", {type: "mouseMoved", x, y});
    await send(tabId, "Input.dispatchMouseEvent", {type: "mousePressed", x, y, button: "left", clickCount: 1});
    await send(tabId, "Input.dispatchMouseEvent", {type: "mouseReleased", x, y, button: "left", clickCount: 1});
  } else if (command === "type_text") {
    if (typeof params.text !== "string" || params.text.length > 100000) throw new Error("Invalid text input.");
    if (params.replace !== false) {
      await press(tabId, navigator.platform.includes("Mac") ? "Meta+A" : "Control+A");
      if (!params.text) await press(tabId, "Backspace");
    }
    if (params.text) await send(tabId, "Input.insertText", {text: params.text});
  } else if (command === "press") {
    await press(tabId, params.key);
  } else if (command === "scroll") {
    if (![params.dx, params.dy].every(value => Number.isFinite(value) && Math.abs(value) <= 100000)) throw new Error("Invalid scroll distance.");
    const info = await describe(tabId);
    await send(tabId, "Input.dispatchMouseEvent", {type: "mouseWheel", x: info.viewport_width / 2, y: info.viewport_height / 2, deltaX: params.dx, deltaY: params.dy});
    await delay(150);
  } else if (command === "navigate") {
    await navigate(tabId, params.url);
  } else {
    throw new Error("Unsupported bridge command.");
  }
  return {};
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === "connect") {
    startConnection(message.config).then(() => sendResponse({ok: true})).catch(async error => {
      await stopConnection(error.message);
      sendResponse({ok: false, error: error.message});
    });
    return true;
  }
  if (message.action === "disconnect") {
    stopConnection().then(() => sendResponse({ok: true}));
    return true;
  }
});

chrome.debugger.onDetach.addListener((source, reason) => {
  const wasAttached = attached.delete(source.tabId);
  if (wasAttached && reason === "canceled_by_user") void stopConnection("Browser control was stopped. Connect again when ready.");
});

async function reconnectSaved() {
  if (running) return;
  const saved = await chrome.storage.local.get("config");
  if (saved.config?.enabled) {
    try { await startConnection(saved.config); }
    catch (error) { await setStatus("Bridge unavailable. Start the CLI and reconnect with its current token. " + error.message); }
  }
}

chrome.alarms.create("groundwork-reconnect", {periodInMinutes: 0.5});
chrome.alarms.onAlarm.addListener(alarm => { if (alarm.name === "groundwork-reconnect") void reconnectSaved(); });
void reconnectSaved();
