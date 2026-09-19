# Local browser CLI

The CLI loads a `.pt` checkpoint produced by Groundwork, captures the selected browser tab, predicts a target from the screenshot and your instruction, and sends mouse or keyboard input. It supports interactive commands and repeatable JSON task files. Inference and orchestration run on this laptop; no hosted planner, API key, or external model service is used. Browsing an online website still makes that website's normal network requests.

The model predicts element locations. You provide the steps; the CLI does not turn an arbitrary goal into a plan. Accuracy depends on your reviewed training data, vocabulary, and held-out results. An unfamiliar website or instruction can produce a wrong target even with a high presence score. Start with `find` to inspect a prediction.

## Start with your trained model

Run from PowerShell in the repository. Existing installations should install the updated requirements once:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
powershell -ExecutionPolicy Bypass -File scripts/agent.ps1 --model "data\exports\YOUR_EXPORT_ID.pt"
```

For a CPU installation use `requirements-cpu-lock.txt` and add `--device cpu`. `--device auto` is the default and uses CUDA when available. The application web server does not need to be running.

For a quick start with `data\exports\test.pt`, let the CLI launch and connect a separate Edge profile:

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --model "data\exports\test.pt" --launch edge --url https://example.com
```

Use `--launch chrome` for installed Chrome. Choose the tab when prompted, then enter commands at `groundwork>`. The CLI enables local debugging and discovers the browser's port automatically; omit `--endpoint` for this path.

Use either an inference export downloaded from Groundwork or a generated training checkpoint under `data\runs\RUN_ID\checkpoints\`. Both contain the model configuration, tokenizer and class mapping. A bare state dictionary or an unrelated `.pt` format is not enough. The source file is never modified. You can also start without `--model` and choose a local model interactively.

Normal generated exports and checkpoints use restricted weight loading. If an older file needs unsupported Python pickle objects, export it again from Groundwork. `--trust-checkpoint` enables an unrestricted pickle fallback only when you explicitly choose it for a file whose source you trust.

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --help
```

## Select an already running browser and website

Use the included extension to connect your normal, already running Chrome or Edge profile and its open tabs:

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --model "data\exports\test.pt"
```

1. Start the CLI without `--endpoint`. Keep its terminal open at `groundwork>`; it prints the loopback bridge address and a session connection token. It can wait here while you connect the extension.
2. In Chrome open `chrome://extensions`, or in Edge open `edge://extensions`. Turn on **Developer mode**, select **Load unpacked**, and select this repository's `browser-extension` folder. See Chrome's [official unpacked extension instructions](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked).
3. Open the Groundwork extension popup, give the browser a recognizable name, enter the bridge address and token printed by the CLI, and connect.
4. In the CLI run `browsers`, then `connect` and choose that browser. Run `tabs`, then `use` and choose the open website by its title, URL, and tab ID.
5. Run `find message field` or another instruction that your model learned. Inspect the prediction and local screenshot before trying `click` or `type`.

Install and connect the extension in each browser/profile you want to select. A new CLI process has a new token; reconnect the extension with that token. Browser input is restricted to the explicitly selected tab. Closing the CLI detaches; your browser and tabs remain open.

Ordinary browsers do not expose their tab control to an arbitrary local process. A running browser must have this extension connected or a local debugging endpoint enabled. Browser internal pages, protected pages, and native dialogs may reject debugger access. If Chrome reports another debugger is attached, close that tab's DevTools or detach the other debugger and retry.

## Alternative: local debugging connection

The CLI also supports a running Chromium-based browser with a local Chrome DevTools Protocol (CDP) endpoint. `--endpoint` only attaches to a browser whose local debugger is already enabled at that address; it does not start a browser or enable debugging in a normally opened window:

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --model "data\exports\YOUR_EXPORT_ID.pt" --endpoint http://127.0.0.1:9222
```

Choose the tab after connecting, or supply `--tab` with the tab ID shown by `tabs`. This implementation accepts loopback endpoints only. See [Playwright's CDP attachment documentation](https://playwright.dev/python/docs/api/class-browsertype#browser-type-connect-over-cdp).

To launch a separate Edge profile at the included practice page, first start the demo server described under [Task files](#task-files), then run:

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --model "data\exports\YOUR_EXPORT_ID.pt" --launch edge --url http://127.0.0.1:8767/browser-demo.html
```

`--launch chrome` uses installed Chrome. `--launch chromium` uses Playwright's browser, which needs the one-time `python -m playwright install chromium` download. A launched browser remains open when the CLI exits, so you can inspect its final state.

Chrome 136 and newer require a non-default `--user-data-dir` when using remote debugging flags. A browser started this way has a separate profile and does not contain your ordinary profile's existing tabs. Use the extension above to select those existing tabs. See [Chrome's remote debugging change](https://developer.chrome.com/blog/remote-debugging-port).

### If no browser responds at the endpoint

Check the CDP address from another PowerShell terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9222/json/version -TimeoutSec 2
```

A working CDP HTTP endpoint returns browser information including `webSocketDebuggerUrl`. A connection error means the CLI cannot attach at that address. Use the `--launch edge` command above, or start without `--endpoint` and connect the extension. The extension bridge (normally port `8766`) and the CDP endpoint (often port `9222`) serve different connections.

If an endpoint is unavailable during interactive startup, the CLI keeps the `groundwork>` prompt and extension bridge open. You can enter `launch edge` or `launch chrome`, or connect the extension and run `browsers`, `connect`, then `use`. Scripted tasks, `--command`, and `--list-tabs` still fail with a connection error when the requested endpoint is unavailable.

If a Triton CUDA-toolkit discovery warning is followed by `Loaded test.pt on cuda`, model loading completed and the browser connection error is a separate failure. The [verification report](verification.md) records the same warning alongside passing CUDA workflows. Check the browser connection first.

## Interactive commands

| Command | Behavior |
| --- | --- |
| `browsers` | List connected extensions and detected local CDP browsers. |
| `connect` / `connect #1` | Choose a browser from the menu or select its numbered choice/ID. |
| `tabs` | List its open websites. |
| `use` / `use #2` | Choose the tab to control; a tab ID or exact unique URL also works. |
| `launch edge` / `launch chrome` | Launch and connect a separate browser profile, then select a tab. |
| `model` / `models` | Choose another generated checkpoint/export or list local model paths. |
| `status` | Show the selected browser, tab, model, and step budget. |
| `screenshot` | Save the selected tab's current visible viewport. |
| `find message field` | Predict a target and save its observation without clicking. |
| `click confirm message` | Capture, predict, and click the predicted center. |
| `type "Hello locally" into message field` | Predict the field, click it, replace its text, and type. |
| `press Enter` | Send a key to the selected tab's current focus. |
| `press Control+A` | Send a key combination. |
| `scroll down 500` | Scroll the selected viewport by 500 CSS pixels. |
| `scroll up 300` | Scroll upward. |
| `open https://example.com` | Navigate the selected tab to an HTTP(S) URL. |
| `wait 1` | Wait one second before the next observation. |
| `run examples/browser-task.json` | Run a validated sequence of steps. |
| `help` / `quit` | Show help / detach and exit. |

Actions request confirmation by default. `--auto` explicitly enables executing your supplied commands or task sequence without per-action confirmation. A task stops when a step fails or the model abstains on an action that requires a target. A presence score is not a calibrated probability that a click will succeed.

Numbered browser and tab choices use `#1`, `#2`, and so on. Bare numeric values are actual browser/tab IDs, because extension tab IDs are numeric. On the command line quote a numbered choice, for example `--tab "#1"`. `--tab "id:123"` explicitly selects ID `123`.

`--threshold 0.7` sets the presence threshold. `--max-steps 50` limits the total session steps, including search scrolls. `--command "find message field"` runs one explicit action and exits; repeat `--command` for a sequence. `--list-models`, `--list-browsers`, and `--list-tabs` print inventories; a browser must already be connected before its tabs can be listed. Press Ctrl+C to stop an interactive sequence.

## Task files

[`examples/browser-task.json`](../examples/browser-task.json) is a complete local practice task. Serve the included page in another terminal:

```powershell
.venv\Scripts\python.exe -m http.server 8767 --bind 127.0.0.1 --directory examples
```

Open `http://127.0.0.1:8767/browser-demo.html` in your chosen browser, connect and select its tab, and run the task. This example needs a model capable of locating the message field and confirmation button; it does not train your model automatically.

Scripted execution can specify an endpoint and tab directly:

```powershell
.venv\Scripts\python.exe -m grounding.agent_cli --model "data\exports\YOUR_EXPORT_ID.pt" --endpoint http://127.0.0.1:9222 --tab "TAB_ID" --task examples/browser-task.json --auto
```

A task has a `name` and a `steps` array:

```json
{
  "name": "Enter a local practice message",
  "steps": [
    { "action": "find", "instruction": "message field" },
    { "action": "type", "instruction": "message field", "text": "Hello locally" },
    { "action": "press", "key": "Enter" },
    { "action": "click", "instruction": "confirm message", "search_scrolls": 2 }
  ]
}
```

Supported actions are `find`, `assert_visible`, `click`, `type`, `press`, `scroll`, `navigate`, and `wait`. Grounded actions take `instruction`; `type` also takes `text` and optionally `replace` (default `true`). `press` takes `key`; `scroll` takes `dx` and `dy`; `navigate` takes `url`; `wait` takes `seconds`. `search_scrolls` enables a bounded search that takes a fresh screenshot after each scroll. `assert_visible` checks model presence, not the DOM or the correctness of a completed transaction. Task files cannot execute JavaScript or shell commands.

Every grounded action starts with a fresh screenshot. Normalized prediction coordinates are converted to CSS viewport pixels, including on high-DPI displays. The CLI checks the selected tab, URL, viewport, and screenshot changes before applying a prediction. This check cannot catch every small change; keep the selected page stable while a step runs.

## Local reports and verification

Each session writes JSONL action records and before/after screenshots under `data/agent_sessions/TIMESTAMP-ID/`. Set `--output-dir` to choose another folder. The action's typed `text` field is omitted from JSONL records. Page titles, URLs, and screenshots can still contain entered text, and task JSON contains any text you put into it. Reports include prediction scores, model vocabulary warnings and failures so you can review and collect corrections. Scripted execution exits with code `0` on completion, `3` when a task stops, `2` for validation/connection errors, and `130` on interruption. If interrupted during an input request, inspect the page and the recorded `outcome_unknown` result before retrying: the browser may already have applied that input.

Run the isolated integration smoke after installing Playwright's test browser once:

```powershell
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe scripts/agent_smoke.py
.venv\Scripts\python.exe scripts/agent_smoke.py --extension
```

The smoke opens its own headless browser and loopback practice page. The default run tests CDP attachment and the actual CLI task entry point; `--extension` loads the real extension, connects through its popup, and tests the same local actions through the authenticated bridge. Both use an explicitly labeled constant-output `.pt` fixture to exercise real loading, screenshot inference, coordinate mapping, input, navigation, abstention, and detaching without closing a selected tab. Add `--device cuda` to test GPU inference or `--output-dir data/agent-smoke-report` to preserve artifacts. This verifies integration only; it is not evidence of learned grounding quality on real websites. It never controls your existing browser tabs.

If a target is missed, collect a screenshot at the same viewport and scale, add reviewed instructions in Groundwork, and evaluate a new training run on independent held-out screenshots. Use short instructions represented in the saved tokenizer. Lowering the threshold merely changes abstention; it does not improve the model's accuracy.
