"""Local browser transports for screenshot-grounded interaction.

The extension relay is bound to 127.0.0.1 and authenticated by a fresh secret.
CDP attachment never creates a context or closes a user's browser or tabs.
There are deliberately no element-selector or arbitrary-JavaScript actions.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hmac
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image


class BrowserError(RuntimeError):
    """An actionable local-browser connection or interaction error."""


def validate_endpoint(endpoint: str) -> str:
    """Accept only literal loopback CDP addresses; normalize localhost to IPv4."""
    try:
        parts = urlsplit(str(endpoint).strip())
        host = parts.hostname
        port = parts.port
        if parts.scheme not in {"http", "ws"} or not host or not port:
            raise ValueError
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError
        if host.lower() == "localhost":
            host = "127.0.0.1"
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError
        if parts.scheme == "http" and parts.path not in {"", "/"}:
            raise ValueError
        if parts.scheme == "ws" and not parts.path.startswith("/devtools/browser/"):
            raise ValueError
        address = f"[{host}]" if ":" in host else host
        return urlunsplit((parts.scheme, f"{address}:{port}", parts.path.rstrip("/"), "", ""))
    except (ValueError, TypeError) as exc:
        raise BrowserError("Use a loopback CDP endpoint such as http://127.0.0.1:9222 (no credentials or redirects).") from exc


def validate_url(url: str, *, allow_blank: bool = False) -> str:
    if allow_blank and url == "about:blank":
        return url
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError
        _ = parts.port
        if any(ord(char) < 32 for char in url):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise BrowserError("Navigation requires an http:// or https:// URL without embedded credentials.") from exc
    return url


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BrowserError("CDP endpoint redirected; only direct loopback connections are allowed.")


_local_http = build_opener(ProxyHandler({}), _NoRedirect())


def _probe_endpoint(endpoint: str, timeout: float = .5) -> dict | None:
    endpoint = validate_endpoint(endpoint)
    if endpoint.startswith("ws:"):
        return {"id": f"cdp:{endpoint}", "name": "Chromium (CDP)", "transport": "cdp", "endpoint": endpoint, "websocket": endpoint}
    try:
        with _local_http.open(Request(endpoint + "/json/version"), timeout=timeout) as response:
            data = json.loads(response.read(256_000))
        websocket = validate_endpoint(data["webSocketDebuggerUrl"])
        if not websocket.startswith("ws:"):
            return None
        browser_name = str(data.get("Browser", "Chromium"))[:120]
        return {"id": f"cdp:{endpoint}", "name": browser_name + " (CDP)", "transport": "cdp", "endpoint": endpoint, "websocket": websocket}
    except (OSError, ValueError, KeyError, TypeError, BrowserError):
        return None


def _process_endpoints() -> set[str]:
    """Read debugging flags of running Chromium processes; never scan a network."""
    found = set()
    try:
        import psutil
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                if not any(name in (process.info.get("name") or "").lower() for name in ("chrome", "chromium", "msedge")):
                    continue
                args = process.info.get("cmdline") or []
                port = None
                profile = None
                for index, arg in enumerate(args):
                    if arg.startswith("--remote-debugging-port="):
                        port = arg.partition("=")[2]
                    elif arg == "--remote-debugging-port" and index + 1 < len(args):
                        port = args[index + 1]
                    elif arg.startswith("--user-data-dir="):
                        profile = arg.partition("=")[2].strip('"')
                    elif arg == "--user-data-dir" and index + 1 < len(args):
                        profile = args[index + 1]
                if port == "0" and profile:
                    try:
                        port = (Path(profile) / "DevToolsActivePort").read_text().splitlines()[0]
                    except (OSError, IndexError, UnicodeError):
                        continue
                if port and port.isdigit() and 0 < int(port) < 65536:
                    found.add(f"http://127.0.0.1:{int(port)}")
            except (psutil.Error, OSError):
                continue
    except ImportError:
        pass
    return found


class _Job:
    def __init__(self, browser_id: str, command: str, params: dict, timeout: float):
        self.id = secrets.token_hex(16)
        self.browser_id = browser_id
        self.command = command
        self.params = params
        self.expires = time.time() + timeout
        self.event = threading.Event()
        self.result: Any = None
        self.error: str | None = None
        self.cancelled = False

    def payload(self) -> dict:
        return {"id": self.id, "command": self.command, "params": self.params, "expires_at": self.expires}


class _RelayServer(ThreadingHTTPServer):
    daemon_threads = True


class BrowserHub:
    def __init__(self, port: int = 8766, *, command_timeout: float = 30):
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise BrowserError("Bridge port must be between 0 and 65535.")
        self.port = port
        self.command_timeout = float(command_timeout)
        self.token = secrets.token_urlsafe(32)
        self.bridge_url = f"http://127.0.0.1:{port}"
        self._server: _RelayServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._extensions: dict[str, dict] = {}
        self._jobs: dict[str, _Job] = {}
        self._known: dict[str, dict] = {}
        self._connections: list[BrowserConnection] = []
        self._playwright = None
        self._closed = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    def start(self):
        if self._server:
            return self
        if self._closed:
            raise BrowserError("This browser hub has been closed; create a new one.")
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass  # Never log authentication headers or screenshots.

            def _send(self, status: int, payload: dict):
                body = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _authorized(self) -> bool:
                # No browser-page CORS permissions; only installed extension hosts.
                origin = self.headers.get("Origin")
                if origin and not re.fullmatch(r"chrome-extension://[a-p]{32}", origin):
                    self._send(403, {"error": "Only a locally installed browser extension may connect."})
                    return False
                if not hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + hub.token).encode("utf-8")):
                    self._send(401, {"error": "Invalid bridge token. Copy the current token from the CLI into the extension."})
                    return False
                return True

            def do_GET(self):
                if not self._authorized():
                    return
                parts = urlsplit(self.path)
                if parts.path == "/v1/status":
                    self._send(200, {"ok": True, "protocol": 1})
                    return
                if parts.path != "/v1/poll":
                    self._send(404, {"error": "Unknown endpoint"})
                    return
                browser_id = parse_qs(parts.query).get("browser_id", [""])[0]
                with hub._lock:
                    client = hub._extensions.get(browser_id)
                    if client:
                        client["last_seen"] = time.monotonic()
                if not client:
                    self._send(404, {"error": "Browser is not registered; reconnect the extension."})
                    return
                deadline = time.monotonic() + 20
                while not hub._closed and time.monotonic() < deadline:
                    try:
                        job = client["queue"].get(timeout=.25)
                    except queue.Empty:
                        continue
                    if job.cancelled or time.time() >= job.expires or job.id not in hub._jobs:
                        continue
                    self._send(200, {"job": job.payload()})
                    return
                self._send(200, {"job": None})

            def do_POST(self):
                if not self._authorized():
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 32 * 1024 * 1024:
                        self._send(413, {"error": "Body is missing or too large."})
                        return
                    self.connection.settimeout(10)
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError
                except (ValueError, OSError, UnicodeError):
                    self._send(400, {"error": "Expected a JSON object."})
                    return
                path = urlsplit(self.path).path
                if path == "/v1/register":
                    instance_id = payload.get("instance_id", "")
                    if not isinstance(instance_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", instance_id):
                        self._send(400, {"error": "Invalid browser instance ID."})
                        return
                    browser_id = "extension:" + instance_id
                    with hub._lock:
                        client = hub._extensions.setdefault(browser_id, {"queue": queue.Queue()})
                        client.update({"id": browser_id, "name": str(payload.get("name") or "Browser extension")[:120], "transport": "extension", "last_seen": time.monotonic()})
                    self._send(200, {"browser_id": browser_id, "protocol": 1})
                elif path == "/v1/result":
                    with hub._lock:
                        job = hub._jobs.get(str(payload.get("job_id")))
                        if not job or job.browser_id != payload.get("browser_id") or job.cancelled:
                            self._send(404, {"error": "Unknown or expired command."})
                            return
                        job.result = payload.get("result")
                        job.error = str(payload["error"])[:2000] if payload.get("error") else None
                        job.event.set()
                    self._send(200, {"ok": True})
                elif path == "/v1/unregister":
                    hub._unregister(str(payload.get("browser_id")))
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "Unknown endpoint"})

        try:
            self._server = _RelayServer(("127.0.0.1", self.port), Handler)
        except OSError as exc:
            raise BrowserError(f"Cannot start local extension bridge on port {self.port}. Close another CLI instance or use --port. {exc}") from exc
        self.port = self._server.server_address[1]
        self.bridge_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": .1}, daemon=True, name="groundwork-browser-relay")
        self._thread.start()
        return self

    def _unregister(self, browser_id: str):
        with self._lock:
            self._extensions.pop(browser_id, None)
            for job in self._jobs.values():
                if job.browser_id == browser_id:
                    job.error = "The browser extension disconnected. Reconnect it in the extension popup."
                    job.event.set()

    def _extension_command(self, browser_id: str, command: str, params: dict | None = None, *, timeout: float | None = None):
        timeout = self.command_timeout if timeout is None else timeout
        with self._lock:
            client = self._extensions.get(browser_id)
            if self._closed or not client or time.monotonic() - client["last_seen"] > 45:
                raise BrowserError("Browser extension is offline. Open its popup and connect using the current CLI token.")
            job = _Job(browser_id, command, params or {}, timeout)
            self._jobs[job.id] = job
            client["queue"].put(job)
        try:
            if not job.event.wait(timeout):
                job.cancelled = True
                raise BrowserError("Browser command timed out. Its outcome may be unknown; inspect the tab before retrying. Check the extension connection.")
            if job.error:
                raise BrowserError(job.error)
            return job.result
        finally:
            # Also cancel on KeyboardInterrupt: an interrupted queued action must
            # never execute later when an extension resumes polling.
            job.cancelled = True
            with self._lock:
                self._jobs.pop(job.id, None)

    def list_browsers(self, endpoints: list[str] | None = None) -> list[dict]:
        self.start()
        with self._lock:
            found = [{key: value for key, value in client.items() if key not in {"queue", "last_seen"}} for client in self._extensions.values() if time.monotonic() - client["last_seen"] <= 45]
        candidates = {validate_endpoint(endpoint) for endpoint in endpoints or []}
        if endpoints is None:
            candidates.update(f"http://127.0.0.1:{port}" for port in (9222, 9223, 9224))
            candidates.update(_process_endpoints())
        candidates.update(item["endpoint"] for item in self._known.values() if item["transport"] == "cdp")
        if candidates:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
                for item in executor.map(_probe_endpoint, sorted(candidates)):
                    if item:
                        # Preserve the launch metadata when refreshing discovery.
                        found.append({**self._known.get(item["id"], {}), **item})
        for item in found:
            self._known[item["id"]] = item
        return found

    def _get_playwright(self):
        if not self._playwright:
            try:
                from playwright.sync_api import sync_playwright
                self._playwright = sync_playwright().start()
            except ImportError as exc:
                raise BrowserError("Playwright is missing. Install requirements.txt (or requirements-cpu.txt).") from exc
        return self._playwright

    def connect(self, browser_id: str) -> BrowserConnection:
        self.start()
        item = self._known.get(browser_id)
        if browser_id.startswith(("http://", "ws://")):
            item = _probe_endpoint(validate_endpoint(browser_id), timeout=2)
        elif browser_id.startswith("cdp:"):
            item = _probe_endpoint(validate_endpoint(browser_id[4:]), timeout=2)
        elif browser_id.startswith("extension:"):
            with self._lock:
                client = self._extensions.get(browser_id)
                if client:
                    item = {"id": browser_id, "name": client["name"], "transport": "extension"}
        if not item:
            raise BrowserError("Browser was not found. Run browsers again or connect the extension using the current token.")
        item = {**self._known.get(item["id"], {}), **item}
        self._known[item["id"]] = item
        connection = BrowserConnection(self, item)
        self._connections.append(connection)
        return connection

    def launch(self, browser: str = "edge", url: str = "about:blank", headless: bool = False) -> dict:
        self.start()
        validate_url(url, allow_blank=True)
        if browser not in {"edge", "chrome", "chromium"}:
            raise BrowserError("Choose edge, chrome, or chromium.")
        executable = None
        if browser == "chromium":
            executable = self._get_playwright().chromium.executable_path
            if not Path(executable).is_file():
                raise BrowserError("Chromium is not installed. Run: python -m playwright install chromium")
        elif sys.platform == "win32":
            suffix = ("Microsoft", "Edge", "Application", "msedge.exe") if browser == "edge" else ("Google", "Chrome", "Application", "chrome.exe")
            for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
                candidate = Path(os.environ.get(variable, ""), *suffix)
                if candidate.is_file():
                    executable = str(candidate)
                    break
        else:
            for name in (("microsoft-edge", "microsoft-edge-stable") if browser == "edge" else ("google-chrome", "google-chrome-stable")):
                executable = shutil.which(name)
                if executable:
                    break
        if not executable:
            raise BrowserError(f"{browser.title()} was not found. Use launch chromium after python -m playwright install chromium.")
        profile = Path(tempfile.mkdtemp(prefix="groundwork-browser-"))
        args = [executable, f"--user-data-dir={profile}", "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "--disable-session-crashed-bubble", "--window-size=1280,800"]
        if headless:
            args.append("--headless=new")
        # Start on a committed blank page, then navigate via CDP below. Passing a
        # URL on Chromium's command line exposes an incompletely initialized
        # target that can deadlock a client's initial frame-tree attachment.
        args.append("about:blank")
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" and headless else 0)
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise BrowserError(f"{browser.title()} exited during startup. Temporary profile: {profile}")
            try:
                port = int(port_file.read_text().splitlines()[0])
                endpoint = f"http://127.0.0.1:{port}"
                item = _probe_endpoint(endpoint, timeout=1)
                # /json/version is exposed before Chromium commits its first
                # page. Attaching Playwright to that empty startup target can
                # race its initial-navigation wait and stall initialization.
                with _local_http.open(Request(endpoint + "/json/list"), timeout=1) as response:
                    targets = json.loads(response.read(256_000))
                ready = any(target.get("type") == "page" and target.get("url") for target in targets)
                if item and ready:
                    item.update({"name": f"{browser.title()} (separate local profile)", "pid": process.pid, "profile_dir": str(profile), "launched": True})
                    self._known[item["id"]] = item
                    if url != "about:blank":
                        connection = self.connect(item["id"])
                        try:
                            tabs = connection.list_tabs()
                            if not tabs:
                                raise BrowserError("The launched browser has no initial tab.")
                            connection.select_tab(tabs[0]["id"])
                            connection.navigate(url)
                        finally:
                            connection.close()
                    return item
            except (OSError, ValueError, IndexError):
                pass
            time.sleep(.1)
        # Only this just-created process can be stopped on a failed launch.
        process.terminate()
        raise BrowserError(f"{browser.title()} did not start its local debugger in time. Temporary profile: {profile}")

    def close(self):
        if self._closed:
            return
        for connection in list(self._connections):
            connection.close()
        self._closed = True
        with self._lock:
            for job in self._jobs.values():
                job.error = "Browser bridge closed."
                job.event.set()
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)


class BrowserConnection:
    def __init__(self, hub: BrowserHub, browser: dict):
        self.hub = hub
        self.browser = browser
        self.transport = browser["transport"]
        self.tab_id: str | None = None
        self._browser = None
        self._page = None
        self._session = None
        self._pages: dict[str, Any] = {}
        self._closed = False
        if self.transport == "cdp":
            try:
                self._browser = hub._get_playwright().chromium.connect_over_cdp(validate_endpoint(browser["websocket"]), timeout=10_000, no_defaults=True)
            except Exception as exc:
                raise BrowserError(f"Cannot attach to the local browser. Confirm remote debugging is enabled: {exc}") from exc

    def _ensure_open(self):
        if self._closed or self.hub._closed:
            raise BrowserError("Browser connection is closed.")

    def _extension(self, command: str, params: dict | None = None):
        self._ensure_open()
        return self.hub._extension_command(self.browser["id"], command, params)

    def _selected(self):
        self._ensure_open()
        if self.tab_id is None:
            raise BrowserError("Select an open website first using tabs and use <tab-id>.")
        if self.transport == "cdp" and (not self._page or self._page.is_closed()):
            raise BrowserError("The selected tab was closed. Run tabs and select another website.")

    def list_tabs(self) -> list[dict]:
        self._ensure_open()
        if self.transport == "extension":
            result = self._extension("list_tabs")
            if not isinstance(result, list):
                raise BrowserError("Invalid tab list received from the extension. Reload the extension.")
            return result
        tabs = []
        self._pages = {}
        try:
            for context in self._browser.contexts:
                for page in context.pages:
                    if page.is_closed():
                        continue
                    session = context.new_cdp_session(page)
                    try:
                        info = session.send("Target.getTargetInfo")["targetInfo"]
                    finally:
                        session.detach()
                    tab_id = str(info["targetId"])
                    self._pages[tab_id] = page
                    tabs.append({"id": tab_id, "title": info.get("title", ""), "url": info.get("url", page.url), "active": tab_id == self.tab_id})
        except Exception as exc:
            raise BrowserError(f"Cannot list browser tabs. The browser may have disconnected: {exc}") from exc
        return tabs

    def select_tab(self, tab_id: str):
        self._ensure_open()
        tab_id = str(tab_id)
        tabs = self.list_tabs()
        if tab_id not in {str(tab["id"]) for tab in tabs}:
            raise BrowserError("That tab is no longer open. Run tabs again and use its current ID.")
        if self.transport == "extension":
            if self.tab_id and self.tab_id != tab_id:
                self._extension("detach", {"tab_id": self.tab_id})
            self._extension("select_tab", {"tab_id": tab_id})
        else:
            if self._session:
                self._session.detach()
            self._page = self._pages[tab_id]
            self._page.set_default_timeout(10_000)
            self._page.bring_to_front()
            self._session = self._page.context.new_cdp_session(self._page)
        self.tab_id = tab_id
        return self.describe()

    def describe(self) -> dict:
        self._selected()
        if self.transport == "extension":
            return self._extension("describe", {"tab_id": self.tab_id})
        try:
            metrics = self._session.send("Page.getLayoutMetrics")
            viewport = metrics.get("cssLayoutViewport", metrics["layoutViewport"])
            info = self._session.send("Target.getTargetInfo")["targetInfo"]
            return {"tab_id": self.tab_id, "url": info.get("url", self._page.url), "title": info.get("title", ""), "viewport_width": viewport["clientWidth"], "viewport_height": viewport["clientHeight"], "scroll_x": viewport.get("pageX", 0), "scroll_y": viewport.get("pageY", 0)}
        except Exception as exc:
            raise BrowserError(f"Cannot inspect the selected tab. Select it again: {exc}") from exc

    def capture(self) -> dict:
        self._selected()
        if self.transport == "extension":
            result = self._extension("capture", {"tab_id": self.tab_id})
        else:
            try:
                result = self.describe()
                # CSS layout dimensions exclude browser scrollbars. Crop exactly
                # that area, otherwise screenshot scaling would shift every click.
                clip = {"x": result["scroll_x"], "y": result["scroll_y"], "width": result["viewport_width"], "height": result["viewport_height"], "scale": 1}
                result["data"] = self._session.send("Page.captureScreenshot", {"format": "png", "fromSurface": True, "captureBeyondViewport": False, "clip": clip})["data"]
            except BrowserError:
                raise
            except Exception as exc:
                raise BrowserError(f"Could not capture the browser viewport: {exc}") from exc
        try:
            png = base64.b64decode(result.pop("data"), validate=True)
            with Image.open(io.BytesIO(png)) as screenshot:
                if screenshot.format != "PNG":
                    raise ValueError("Expected PNG")
                width, height = screenshot.size
            if width <= 0 or height <= 0 or result["viewport_width"] <= 0 or result["viewport_height"] <= 0:
                raise ValueError("Empty viewport")
            return {**result, "png": png, "width": width, "height": height}
        except (KeyError, ValueError, OSError, TypeError) as exc:
            raise BrowserError("The browser returned an invalid viewport screenshot.") from exc

    def click(self, x: float, y: float):
        self._selected()
        info = self.describe()
        if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (x, y)) or not 0 <= x < info["viewport_width"] or not 0 <= y < info["viewport_height"]:
            raise BrowserError("Click coordinates must be finite CSS pixels inside the selected viewport.")
        if self.transport == "extension":
            return self._extension("click", {"tab_id": self.tab_id, "x": x, "y": y})
        try:
            self._page.mouse.click(x, y)
        except Exception as exc:
            raise BrowserError(f"Browser click failed: {exc}") from exc

    def type_text(self, text: str, replace: bool = True):
        self._selected()
        if not isinstance(text, str) or len(text) > 100_000:
            raise BrowserError("Text must be a string of at most 100,000 characters.")
        if self.transport == "extension":
            return self._extension("type_text", {"tab_id": self.tab_id, "text": text, "replace": bool(replace)})
        try:
            if replace:
                self._page.keyboard.press("Meta+A" if sys.platform == "darwin" else "Control+A")
                if not text:
                    self._page.keyboard.press("Backspace")
            if text:
                self._page.keyboard.insert_text(text)
        except Exception as exc:
            raise BrowserError(f"Browser typing failed: {exc}") from exc

    def press(self, key: str):
        self._selected()
        if not isinstance(key, str) or not key or len(key) > 80:
            raise BrowserError("Supply a keyboard key such as Enter, Tab, or Control+A.")
        if self.transport == "extension":
            return self._extension("press", {"tab_id": self.tab_id, "key": key})
        try:
            parts = key.split("+")
            final = parts[-1]
            extra_function_key = bool(re.fullmatch(r"F(1[3-9]|2[0-4])", final))
            if extra_function_key or (len(final) == 1 and ord(final) > 126):
                # Playwright's physical-key map omits Unicode and F13–F24.
                # CDP accepts these directly, also used by the extension.
                modifiers = 0
                for modifier in parts[:-1]:
                    if modifier == "ControlOrMeta":
                        modifier = "Meta" if sys.platform == "darwin" else "Control"
                    bit = {"Alt": 1, "Control": 2, "Meta": 4, "Shift": 8}.get(modifier)
                    if bit is None:
                        raise BrowserError("Use canonical keyboard modifiers such as Control or Shift.")
                    modifiers |= bit
                text = "" if extra_function_key else final.upper() if modifiers & 8 else final
                if modifiers & 7:
                    text = ""
                params = {"key": final, "modifiers": modifiers}
                if extra_function_key:
                    params.update({"code": final, "windowsVirtualKeyCode": 111 + int(final[1:])})
                self._session.send("Input.dispatchKeyEvent", {**params, "type": "keyDown" if text else "rawKeyDown", **({"text": text} if text else {})})
                self._session.send("Input.dispatchKeyEvent", {**params, "type": "keyUp"})
            else:
                self._page.keyboard.press(key)
        except Exception as exc:
            raise BrowserError(f"Browser key press failed: {exc}") from exc

    def scroll(self, dx: float, dy: float):
        self._selected()
        if not all(isinstance(value, (float, int)) and math.isfinite(value) and abs(value) <= 100_000 for value in (dx, dy)):
            raise BrowserError("Scroll distances must be finite numbers up to 100,000 CSS pixels.")
        if self.transport == "extension":
            return self._extension("scroll", {"tab_id": self.tab_id, "dx": dx, "dy": dy})
        try:
            info = self.describe()
            self._page.mouse.move(info["viewport_width"] / 2, info["viewport_height"] / 2)
            self._page.mouse.wheel(dx, dy)
            self._page.wait_for_timeout(150)
        except Exception as exc:
            raise BrowserError(f"Browser scroll failed: {exc}") from exc

    def navigate(self, url: str):
        self._selected()
        validate_url(url)
        if self.transport == "extension":
            return self._extension("navigate", {"tab_id": self.tab_id, "url": url})
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=25_000)
        except Exception as exc:
            raise BrowserError(f"Navigation failed or timed out; inspect the selected tab before retrying: {exc}") from exc

    def close(self):
        if self._closed:
            return
        try:
            if self.transport == "extension" and self.tab_id:
                self.hub._extension_command(self.browser["id"], "detach", {"tab_id": self.tab_id}, timeout=2)
            elif self._browser:
                if self._session:
                    self._session.detach()
                # For a browser connected over CDP, close disconnects the client.
                # No browser/context/page is ever launched through Playwright here.
                self._browser.close()
        except Exception:
            pass
        finally:
            self._closed = True
            self._page = self._session = self._browser = None
