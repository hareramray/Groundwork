# Capture webpages for training

**Groundwork Dataset Capture** collects a webpage at different viewport sizes. It changes the real page viewport so responsive layouts render at each size, then saves exact-size PNGs and their metadata in a ZIP. Capture works locally without starting the lab, running the browser CLI, or loading a model.

## Install in Chrome or Edge

1. Open `chrome://extensions` in Chrome, or **Extensions → Manage extensions** in Edge.
2. Turn on **Developer mode** and choose **Load unpacked**.
3. Select this repository's **`capture-extension`** folder, containing `manifest.json`.
4. Pin **Groundwork Dataset Capture** in the browser's Extensions menu.

These are the browsers' documented local installation flows: [Chrome](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked), [Edge](https://learn.microsoft.com/en-us/microsoft-edge/extensions/getting-started/extension-sideloading). When updating the extension source, use **Reload** on its extension card and reopen its capture page.

The manifest requires Chromium 120 or later. Its `tabs` permission lists open website tabs. Its `debugger` permission temporarily controls the selected tab's viewport and captures PNGs through Chrome's [debugger API](https://developer.chrome.com/docs/extensions/reference/api/debugger). During a capture, the browser may display a debugging banner. Capture data stays in the extension's local browser storage until you export or delete it.

## Collect screenshots

1. Open the webpage you want to collect. Finish any login, navigation, or interaction needed to show the desired page state.
2. Click the extension icon. Its capture workspace opens in a separate tab with your webpage selected under **Source tab**. **Refresh tabs** updates the list if you open more pages.
3. Enter a **Capture name** and **Dataset group**. The group defaults to the page's origin and path. Reuse a broader website, template-family, or collection-session group for related pages so their examples remain together in training splits.
4. Select desktop, tablet, and phone viewport presets, or add a custom width and height. The default selection is 1366 × 768, 768 × 1024, and 390 × 844.
5. Click **Capture selected sizes**. Keep the source tab open and avoid interacting with it until capture finishes. The extension scrolls to the top for each snapshot, waits for the configured **Settle delay**, and captures one viewport. It restores the original viewport and scroll position when it releases the tab.
6. Inspect the thumbnails, then click **Export capture ZIP**. Each ZIP contains PNG files and a `manifest.json` describing their page URL, title, dimensions, timestamp, scroll offset, and dataset group.

You can cancel a batch and export the snapshots already saved. Select earlier batches through **Saved capture batches**; they survive closing and reopening the capture workspace. **Delete batch** frees its saved screenshots from extension storage; previously downloaded ZIPs and imported lab images remain available.

Each side must be 240–4096 pixels. A batch supports up to eight unique sizes and 40 million pixels in total. Captures use device scale factor 1, so CSS viewport dimensions and PNG dimensions match. Phone and tablet presets exercise responsive widths; they do not emulate a mobile user agent, touch input, or a physical device. This version captures the top viewport, not stitched full pages.

## Import and train

Start the lab when you are ready to annotate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

1. Open **http://127.0.0.1:8000 → Images & annotations**.
2. Choose **Import capture ZIP** and select the exported ZIP.
3. Open each imported screenshot, draw target boxes, assign classes, and write instructions for what the model should locate.
4. Explicitly review your annotations, create a dataset version, and train or retrain your grounding model.

Imported snapshots begin **unannotated** and retain their dataset group and capture metadata. They need reviewed instruction/target examples before they can enter a training snapshot. Importing the same capture ZIP again creates additional images. The ZIP can also be extracted for use in another annotation tool; see the [capture bundle format](dataset-format.md#webpage-capture-bundles).

## Troubleshooting

- **Cannot capture this tab:** choose a normal HTTP/HTTPS webpage. Close DevTools or disconnect another debugger from the source tab, including an active Groundwork browser CLI session. Browser settings, extension pages, and pages blocked by browser policy cannot be captured.
- **Page changes or closes during capture:** the batch stops and retains completed screenshots. Reopen or finish navigating the source page, then start a new batch.
- **Images or fonts have not appeared:** wait for the source page to load, or increase **Settle delay** (250–10,000 ms). Captures record the rendered page at that moment; live content and animations may differ between sizes.
- **Capture interrupted by a browser restart:** reopen the capture workspace. Completed images remain exportable, and the interrupted batch is marked as an error. Start a new batch for missing sizes.
- **Storage is full:** export the batches you want to keep, then delete old batches or capture fewer/smaller sizes. The extension uses local IndexedDB for PNGs.
- **Import capture ZIP is missing:** run `npm run build` inside `frontend`, then restart the lab using the start script. Use **Images & annotations** for capture ZIPs; **Dataset versions** imports previously annotated dataset exports.

## Developer verification

The extension is plain JavaScript and needs no build step. To run its tests from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_captures.py
node --test tests/extension_capture_core.mjs
Push-Location frontend
npm run build
npx playwright install chromium
Pop-Location
node tests/extension_capture.mjs
```

The integration runner installs the real unpacked extension in an isolated Chromium profile, serves a generated responsive page, captures different CSS breakpoints, checks PNG pixels and dimensions, verifies page restoration, exports a ZIP, and imports it through the lab. Artifacts go under ignored `test-results/`; the runner does not use your normal browser profile or dataset.
