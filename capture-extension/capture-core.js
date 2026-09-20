// The capture engine has no browser globals other than standard Web APIs so its
// lifecycle and cleanup can be tested with a deterministic CDP adapter.
export function validateSettings(input) {
  if (!input || typeof input !== "object") throw new Error("Capture settings are required.");
  if (!Number.isSafeInteger(input.tabId) || input.tabId < 1) throw new Error("Select a website tab to capture.");
  if (!Array.isArray(input.resolutions) || !input.resolutions.length) throw new Error("Choose at least one viewport size.");
  const resolutions = [];
  const seen = new Set();
  for (const value of input.resolutions) {
    if (!value || ![value.width, value.height].every(number => Number.isInteger(number) && number >= 240 && number <= 4096)) {
      throw new Error("Viewport width and height must be whole numbers between 240 and 4096 pixels.");
    }
    const key = `${value.width}x${value.height}`;
    if (!seen.has(key)) { seen.add(key); resolutions.push({width: value.width, height: value.height}); }
  }
  if (resolutions.length > 8) throw new Error("Choose at most eight unique viewport sizes per batch.");
  if (resolutions.reduce((sum, size) => sum + size.width * size.height, 0) > 40000000) {
    throw new Error("A batch can contain at most 40 million pixels. Choose fewer or smaller viewports.");
  }
  const delayMs = input.delayMs === undefined ? 750 : input.delayMs;
  if (!Number.isInteger(delayMs) || delayMs < 250 || delayMs > 10000) {
    throw new Error("The rendering delay must be between 250 and 10000 milliseconds.");
  }
  for (const key of ["name", "group"]) {
    if (input[key] !== undefined && typeof input[key] !== "string") throw new Error(`The batch ${key} must be text.`);
  }
  return {tabId: input.tabId, name: (input.name || "Website captures").trim().slice(0, 160) || "Website captures",
    group: (input.group || "").trim().slice(0, 500), resolutions, delayMs};
}

export function pngDimensions(base64) {
  if (typeof base64 !== "string" || base64.length < 32) throw new Error("The browser returned an invalid PNG screenshot.");
  let bytes;
  try { bytes = Uint8Array.from(atob(base64.slice(0, 32)), character => character.charCodeAt(0)); }
  catch { throw new Error("The browser returned an invalid PNG screenshot."); }
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (signature.some((value, index) => bytes[index] !== value) ||
      String.fromCharCode(...bytes.slice(12, 16)) !== "IHDR") throw new Error("The browser returned an invalid PNG screenshot.");
  const view = new DataView(bytes.buffer);
  return {width: view.getUint32(16), height: view.getUint32(20)};
}

function abortReason(signal) {
  return signal?.reason instanceof Error ? signal.reason : new DOMException("Capture canceled.", "AbortError");
}

function checkAbort(signal) { if (signal?.aborted) throw abortReason(signal); }

async function waitForRender(adapter, milliseconds, signal) {
  checkAbort(signal);
  if (!signal) return adapter.sleep(milliseconds);
  let listener;
  const canceled = new Promise((_, reject) => {
    listener = () => reject(abortReason(signal));
    signal.addEventListener("abort", listener, {once: true});
  });
  try { await Promise.race([adapter.sleep(milliseconds), canceled]); }
  finally { signal.removeEventListener("abort", listener); }
  checkAbort(signal);
}

function viewport(metrics) {
  const value = metrics?.cssLayoutViewport || metrics?.layoutViewport;
  if (!value || ![value.clientWidth, value.clientHeight, value.pageX, value.pageY].every(Number.isFinite)) {
    throw new Error("The browser could not measure this page's viewport.");
  }
  return value;
}

function scrollCommand(x, y) {
  return {expression: `window.scrollTo({left:${x},top:${y},behavior:'instant'});`, returnByValue: true};
}

export async function runCapture({settings, batch, adapter, store, signal, onUpdate = () => {}}) {
  settings = validateSettings(settings);
  const tabId = settings.tabId;
  let attached = false;
  let overridden = false;
  let original;
  const now = adapter.now || (() => new Date().toISOString());
  const uuid = adapter.uuid || (() => crypto.randomUUID());
  const update = async () => { await store.saveBatch(batch); await onUpdate(batch); };
  batch.captures ||= [];
  batch.errors ||= [];
  try {
    checkAbort(signal);
    await adapter.attach(tabId);
    attached = true;
    checkAbort(signal);
    await adapter.send(tabId, "Page.enable", {});
    original = viewport(await adapter.send(tabId, "Page.getLayoutMetrics", {}));
    batch.original_scroll = {x: original.pageX, y: original.pageY};
    await update();
    for (const {width, height} of settings.resolutions) {
      checkAbort(signal);
      batch.message = `Capturing ${width} × ${height} (${batch.captures.length + 1}/${settings.resolutions.length})`;
      await update();
      overridden = true;
      await adapter.send(tabId, "Emulation.setDeviceMetricsOverride", {
        width, height, deviceScaleFactor: 1, mobile: false, screenWidth: width, screenHeight: height,
      });
      checkAbort(signal);
      await adapter.send(tabId, "Runtime.evaluate", scrollCommand(0, 0));
      await waitForRender(adapter, settings.delayMs, signal);
      const measured = viewport(await adapter.send(tabId, "Page.getLayoutMetrics", {}));
      if (measured.clientWidth !== width || measured.clientHeight !== height) {
        throw new Error(`The page viewport is ${measured.clientWidth} × ${measured.clientHeight}, not the requested ${width} × ${height}.`);
      }
      checkAbort(signal);
      const image = await adapter.send(tabId, "Page.captureScreenshot", {
        format: "png", fromSurface: true, captureBeyondViewport: false,
        clip: {x: measured.pageX, y: measured.pageY, width, height, scale: 1},
      });
      checkAbort(signal);
      const dimensions = pngDimensions(image?.data);
      if (dimensions.width !== width || dimensions.height !== height) {
        throw new Error(`The browser returned a ${dimensions.width} × ${dimensions.height} screenshot for a ${width} × ${height} viewport.`);
      }
      const id = uuid();
      await store.saveImage(id, `data:image/png;base64,${image.data}`);
      batch.captures.push({id, image_path: `images/${id}-${width}x${height}.png`, width, height,
        viewport_width: measured.clientWidth, viewport_height: measured.clientHeight, device_scale_factor: 1,
        url: batch.source_url, title: batch.title, captured_at: now(), scroll_x: measured.pageX, scroll_y: measured.pageY});
      batch.completed = batch.captures.length;
      await update();
    }
    checkAbort(signal);
    batch.status = "completed";
    batch.message = `Saved ${batch.captures.length} screenshot${batch.captures.length === 1 ? "" : "s"}.`;
  } catch (error) {
    const canceled = signal?.aborted && abortReason(signal).name === "AbortError";
    batch.status = canceled ? "canceled" : "error";
    batch.message = canceled ? "Capture canceled. Completed screenshots are saved." : (error.message || String(error));
    if (!canceled) batch.errors.push(batch.message);
  } finally {
    if (attached) {
      const failures = [];
      if (overridden) {
        try { await adapter.send(tabId, "Emulation.clearDeviceMetricsOverride", {}); }
        catch (error) { failures.push(`Restore viewport: ${error.message || error}`); }
      }
      if (original) {
        try { await adapter.send(tabId, "Runtime.evaluate", scrollCommand(original.pageX, original.pageY)); }
        catch (error) { failures.push(`Restore scroll: ${error.message || error}`); }
      }
      try { await adapter.detach(tabId); }
      catch (error) { failures.push(`Release tab: ${error.message || error}`); }
      if (failures.length) {
        batch.errors.push(...failures);
        if (batch.status === "completed") {
          batch.status = "error";
          batch.message = "Screenshots were saved, but the browser could not fully restore the selected tab.";
        }
      }
    }
    batch.completed = batch.captures.length;
    batch.finished_at = now();
    await update();
  }
  return batch;
}
