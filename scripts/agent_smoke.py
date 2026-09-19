"""Isolated browser/PT integration check, not learned-model accuracy evidence.

The only page controlled is the bundled practice page in a disposable browser.
DOM reads below are test assertions, never target selection for the agent.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass


class BrowserObserver:
    """Keep the assertion-only Playwright driver on its own thread.

    Playwright cannot start two synchronous event loops on the same thread.
    The agent owns its own independent driver on the main thread.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.worker = ThreadPoolExecutor(max_workers=1)

    def call(self, action):
        return self.worker.submit(action, self).result(timeout=60)

    def __enter__(self):
        def start(state):
            from playwright.sync_api import sync_playwright
            state.playwright = sync_playwright().start()
            state.browser = state.playwright.chromium.connect_over_cdp(state.endpoint, no_defaults=True)
            state.context = state.browser.contexts[0]
        self.call(start)
        return self

    def __exit__(self, *args):
        def stop(state):
            state.browser.close()
            state.playwright.stop()
        try:
            self.call(stop)
        finally:
            self.worker.shutdown(wait=True)


@contextmanager
def demo_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "examples")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/browser-demo.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def write_fixture(path: Path) -> None:
    """Save a real Groundwork export whose weights always predict the center."""
    import torch
    from grounding.ml import ARCHITECTURE, PREPROCESSING, Grounder, Tokenizer
    from grounding.training import CHECKPOINT_SCHEMA, default_config

    config = {**default_config(), "width": 8, "text_dim": 16, "image_size": 32, "device": "cpu"}
    tokenizer = Tokenizer.build(["message field", "confirm message"])
    model = Grounder(len(tokenizer.vocabulary), 2, config)
    epsilon = 1e-4
    x1, y1, x2, y2 = .40, .46, .60, .54
    fractions = [x1 / (1 - 2 * epsilon), y1 / (1 - 2 * epsilon),
                 (x2 - x1 - epsilon) / (1 - x1 - epsilon),
                 (y2 - y1 - epsilon) / (1 - y1 - epsilon)]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.box_head.bias.copy_(torch.tensor([math.log(x / (1 - x)) for x in fractions]))
        model.presence_head.bias.fill_(8)
    torch.save({"schema": CHECKPOINT_SCHEMA, "kind": "inference_export", "architecture": ARCHITECTURE,
                "preprocessing": PREPROCESSING, "model": model.state_dict(), "config": config,
                "tokenizer": tokenizer.to_dict(), "classes": ["input", "button"],
                "fixture_description": "Constant-output integration fixture; not trained and not quality evidence"}, path)


@contextmanager
def isolated_chromium(executable: str, directory: Path, extension: bool = False):
    import psutil

    port = unused_port()
    endpoint = f"http://127.0.0.1:{port}"
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    extension_flags = ([f"--disable-extensions-except={ROOT / 'browser-extension'}",
                        f"--load-extension={ROOT / 'browser-extension'}"] if extension else [])
    process = subprocess.Popen([executable, "--headless=new", "--no-sandbox", "--no-first-run",
                                "--disable-background-networking", "--force-device-scale-factor=2", "--window-size=1280,720",
                                "--remote-debugging-address=127.0.0.1",
                                f"--remote-debugging-port={port}", f"--user-data-dir={directory}",
                                *extension_flags, "about:blank"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               **options)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Disposable Chromium exited with status {process.returncode}")
            try:
                with urlopen(f"{endpoint}/json/version", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("Disposable Chromium did not expose its loopback CDP endpoint")
        yield endpoint
    finally:
        # Only terminate the subprocess tree created by this smoke check.
        try:
            own_process = psutil.Process(process.pid)
            owned = own_process.children(recursive=True) + [own_process]
        except psutil.NoSuchProcess:
            owned = []
        for child in reversed(owned):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, surviving = psutil.wait_procs(owned, timeout=5)
        for child in surviving:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=10)


def ensure(result: dict, *statuses: str) -> dict:
    if result.get("status") not in statuses:
        raise AssertionError(f"Unexpected agent result: {result}")
    return result


def run_smoke(directory: Path, report_dir: Path, device: str) -> dict:
    from playwright.sync_api import sync_playwright
    from PIL import Image, ImageChops, ImageStat
    from grounding.agent_engine import SessionEngine
    from grounding.agent_model import FileGrounder
    from grounding.browser_runtime import BrowserHub

    model_path = directory / "constant-center-INTEGRATION-FIXTURE.pt"
    write_fixture(model_path)
    model = FileGrounder(model_path, device=device)
    checks = []
    with sync_playwright() as browser_paths:
        executable = browser_paths.chromium.executable_path
    if not Path(executable).exists():
        raise RuntimeError("Install the isolated test browser: .venv\\Scripts\\python.exe -m playwright install chromium")
    with demo_server() as demo_url, isolated_chromium(executable, directory / "browser-profile") as endpoint:
        with BrowserObserver(endpoint) as observer:
            def prepare(state):
                state.unrelated = state.context.new_page()
                state.unrelated.goto(demo_url + "?unrelated=1")
                state.page = state.context.new_page()
                state.page.goto(demo_url)
            observer.call(prepare)
            with BrowserHub(port=unused_port()) as hub:
                candidates = hub.list_browsers(endpoints=[endpoint])
                selected = next(item for item in candidates if item.get("endpoint", "").rstrip("/") == endpoint)
                connection = hub.connect(selected["id"])
                tab = next(item for item in connection.list_tabs() if item["url"] == demo_url)
                connection.select_tab(tab["id"])
                assert connection.describe()["url"] == demo_url
                checks.append("browser and explicit tab selection")

                capture = connection.capture()
                dimensions = {key: value for key, value in capture.items() if key != "png"}
                assert capture["width"] == capture["viewport_width"] * 2, dimensions
                assert capture["height"] == capture["viewport_height"] * 2, dimensions
                engine = SessionEngine(connection, model=model, output_dir=report_dir, auto=True,
                                       threshold=.5, max_steps=30, settle_seconds=.1)
                ensure(engine.execute({"action": "find", "instruction": "message field"}), "found")
                ensure(engine.execute({"action": "type", "instruction": "message field",
                                       "text": "local smoke message"}), "executed")
                assert observer.call(lambda state: state.page.locator("#message").input_value()) == "local smoke message"
                checks.append("real saved PT inference and typing at 2x screenshot scale")

                ensure(engine.execute({"action": "press", "key": "Enter"}), "executed")
                assert observer.call(lambda state: state.page.evaluate("window.demoState.submitted"))
                ensure(engine.execute({"action": "click", "instruction": "confirm message"}), "executed")
                assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 1
                checks.append("keyboard submission and model-grounded click")

                engine.threshold = 1.0
                ensure(engine.execute({"action": "click", "instruction": "confirm message"}), "abstained", "blocked")
                assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 1
                engine.threshold = .5
                checks.append("abstention prevents an additional click")

                before_scroll = connection.capture()
                ensure(engine.execute({"action": "scroll", "dx": 0, "dy": 480}), "executed")
                observer.call(lambda state: state.page.wait_for_function("window.scrollY > 0"))
                after_scroll = connection.capture()
                assert after_scroll["width"] == after_scroll["viewport_width"] * 2
                assert after_scroll["height"] == after_scroll["viewport_height"] * 2
                # The fixed-position button must remain at the same image location
                # after scrolling; catches incorrect screenshot clipping offsets.
                with Image.open(io.BytesIO(before_scroll["png"])) as first, Image.open(io.BytesIO(after_scroll["png"])) as second:
                    crop = tuple(round(value) for value in (first.width * .3, first.height * .4,
                                                            first.width * .7, first.height * .6))
                    difference = ImageChops.difference(first.crop(crop).convert("RGB"), second.crop(crop).convert("RGB"))
                    assert max(ImageStat.Stat(difference).mean) < 1, "Scrolled screenshot lost the fixed-position target"
                ensure(engine.execute({"action": "click", "instruction": "confirm message"}), "executed")
                assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 2
                checks.append("viewport scrolling, screenshot crop, and fresh target click")
                ensure(engine.execute({"action": "navigate", "url": demo_url + "?navigated=1"}), "executed")
                observer.call(lambda state: state.page.wait_for_url(demo_url + "?navigated=1"))
                assert observer.call(lambda state: state.page.url) == demo_url + "?navigated=1"
                ensure(engine.execute({"action": "wait", "seconds": .05}), "executed")
                assert observer.call(lambda state: state.unrelated.url) == demo_url + "?unrelated=1"
                assert observer.call(lambda state: state.unrelated.evaluate("window.demoState.clicks")) == 0
                checks.append("navigation affects only the selected tab")
                connection.close()

            assert observer.call(lambda state: not state.page.is_closed() and not state.unrelated.is_closed())
            assert observer.call(lambda state: state.page.title()) == "Groundwork browser practice"
            checks.append("disconnect preserves selected and unrelated tabs")

            # Verify the actual CLI entry point, argument parsing, and task path.
            task_path = directory / "cli-task.json"
            task = json.loads((ROOT / "examples" / "browser-task.json").read_text(encoding="utf-8"))
            task["steps"][0]["url"] = demo_url
            task_path.write_text(json.dumps(task), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, "-m", "grounding.agent_cli", "--model", str(model_path), "--device", device,
                "--endpoint", endpoint, "--tab", str(tab["id"]), "--task", str(task_path), "--auto",
                "--port", str(unused_port()), "--output-dir", str(report_dir / "cli"),
            ], cwd=ROOT, capture_output=True, text=True, timeout=120)
            assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
            assert observer.call(lambda state: state.page.evaluate("window.demoState.typed")) == "Hello from my local model"
            assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 1
            assert observer.call(lambda state: not state.page.is_closed() and not state.unrelated.is_closed())
            checks.append("unattended CLI JSON task and clean detach")
            assert list(report_dir.rglob("*.png")), "Expected before/after screenshots"
            logs = list(report_dir.rglob("*.jsonl"))
            assert logs, "Expected local JSONL action reports"
            for path in logs:
                assert "local smoke message" not in path.read_text(encoding="utf-8")
                assert "Hello from my local model" not in path.read_text(encoding="utf-8")
            checks.append("local screenshots and redacted typed-text logs")
    return {"status": "passed", "fixture": "constant-output integration model; not learned accuracy evidence",
            "device": device, "checks": checks}


def run_extension_smoke(directory: Path, report_dir: Path, device: str) -> dict:
    """Load the real MV3 extension in the disposable browser and use its relay."""
    from playwright.sync_api import sync_playwright
    from grounding.agent_engine import SessionEngine
    from grounding.agent_model import FileGrounder
    from grounding.browser_runtime import BrowserHub

    model_path = directory / "constant-center-INTEGRATION-FIXTURE.pt"
    write_fixture(model_path)
    model = FileGrounder(model_path, device=device)
    with sync_playwright() as browser_paths:
        executable = browser_paths.chromium.executable_path
    if not Path(executable).exists():
        raise RuntimeError("Install the isolated test browser: .venv\\Scripts\\python.exe -m playwright install chromium")
    with demo_server() as demo_url, isolated_chromium(executable, directory / "browser-profile", extension=True) as endpoint:
        with BrowserObserver(endpoint) as observer, BrowserHub(port=unused_port()) as hub:
            def prepare(state):
                state.unrelated = state.context.new_page()
                state.unrelated.goto(demo_url + "?unrelated=1")
                state.page = state.context.new_page()
                state.page.goto(demo_url)
                deadline = time.monotonic() + 15
                worker = None
                expected_name = json.loads((ROOT / "browser-extension" / "manifest.json").read_text(encoding="utf-8"))["name"]
                while time.monotonic() < deadline:
                    for candidate in state.context.service_workers:
                        if candidate.evaluate("chrome.runtime.getManifest().name") == expected_name:
                            worker = candidate
                            break
                    if worker:
                        break
                    state.page.wait_for_timeout(100)
                if worker is None:
                    raise RuntimeError("The local Groundwork extension did not load into isolated Chromium")
                extension_id = worker.url.split("/")[2]
                state.popup = state.context.new_page()
                state.popup.goto(f"chrome-extension://{extension_id}/popup.html")
                state.popup.locator("#name").fill("Isolated extension smoke")
                state.popup.locator("#url").fill(hub.bridge_url)
                state.popup.locator("#token").fill(hub.token)
                state.popup.locator("#connect").click()
                state.popup.wait_for_function("document.getElementById('status').textContent.includes('Connected')", timeout=15_000)
            observer.call(prepare)
            browser = next(item for item in hub.list_browsers() if item["transport"] == "extension")
            connection = hub.connect(browser["id"])
            tabs = connection.list_tabs()
            tab = next(item for item in tabs if item["url"] == demo_url)
            connection.select_tab(tab["id"])
            capture = connection.capture()
            assert capture["width"] == capture["viewport_width"] * 2
            assert capture["height"] == capture["viewport_height"] * 2
            engine = SessionEngine(connection, model=model, output_dir=report_dir, auto=True,
                                   threshold=.5, max_steps=30, settle_seconds=.1)
            ensure(engine.execute({"action": "find", "instruction": "message field"}), "found")
            ensure(engine.execute({"action": "type", "instruction": "message field", "text": "replace this"}), "executed")
            ensure(engine.execute({"action": "press", "key": "Control+A"}), "executed")
            ensure(engine.execute({"action": "press", "key": "Backspace"}), "executed")
            assert observer.call(lambda state: state.page.locator("#message").input_value()) == ""
            ensure(engine.execute({"action": "type", "instruction": "message field", "text": "Local extension smoke"}), "executed")
            ensure(engine.execute({"action": "press", "key": "Enter"}), "executed")
            ensure(engine.execute({"action": "click", "instruction": "confirm message"}), "executed")
            assert observer.call(lambda state: state.page.evaluate("window.demoState.typed")) == "Local extension smoke"
            assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 1
            engine.threshold = 1.0
            ensure(engine.execute({"action": "click", "instruction": "confirm message"}), "abstained", "blocked")
            assert observer.call(lambda state: state.page.evaluate("window.demoState.clicks")) == 1
            engine.threshold = .5
            ensure(engine.execute({"action": "scroll", "dx": 0, "dy": 400}), "executed")
            observer.call(lambda state: state.page.wait_for_function("window.scrollY > 0"))
            ensure(engine.execute({"action": "navigate", "url": demo_url + "?extension=1"}), "executed")
            observer.call(lambda state: state.page.wait_for_url(demo_url + "?extension=1"))
            assert observer.call(lambda state: state.unrelated.evaluate("window.demoState.clicks")) == 0
            connection.close()
            observer.call(lambda state: state.popup.bring_to_front())
            observer.call(lambda state: state.popup.locator("#disconnect").click())
            assert observer.call(lambda state: not state.page.is_closed() and not state.unrelated.is_closed())
            assert observer.call(lambda state: state.page.title()) == "Groundwork browser practice"
            logs = list(report_dir.rglob("*.jsonl"))
            assert logs and list(report_dir.rglob("*.png"))
            assert all("Local extension smoke" not in path.read_text(encoding="utf-8") for path in logs)
    return {"status": "passed", "fixture": "constant-output integration model; not learned accuracy evidence",
            "device": device, "transport": "real loaded browser extension",
            "checks": ["popup token connection", "running browser and tab listing/selection", "2x screenshot scaling",
                       "real PT inference and typing", "Control+A and Backspace", "Enter and model-grounded click",
                       "abstention prevents click", "scroll and navigation", "unrelated tab preserved",
                       "extension disconnect keeps both tabs open", "local screenshots and redacted logs"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--extension", action="store_true", help="Test the actual loaded extension instead of CDP/CLI attachment")
    parser.add_argument("--output-dir", type=Path, help="Keep local reports in this folder (default: temporary)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="groundwork-agent-smoke-") as temporary:
        directory = Path(temporary)
        report_dir = args.output_dir.resolve() if args.output_dir else directory / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        smoke = run_extension_smoke if args.extension else run_smoke
        result = smoke(directory, report_dir, args.device)
        if args.output_dir:
            result["reports"] = str(report_dir)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
