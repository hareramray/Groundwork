"""Browser transport tests use only a disposable browser and loopback demo."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import threading
from urllib.request import urlopen

from PIL import Image
import psutil
import pytest

from grounding.browser_runtime import BrowserError, BrowserHub, validate_endpoint, validate_url


@pytest.mark.parametrize("endpoint", [
    "https://example.com:9222", "http://example.com:9222", "http://192.168.1.4:9222",
    "http://0.0.0.0:9222", "http://127.0.0.1.evil.test:9222", "http://user:pass@127.0.0.1:9222",
    "http://127.0.0.1:9222/path", "http://127.0.0.1:9222?token=abc", "http://127.0.0.1:99999",
    "ws://127.0.0.1:9222/other", "file:///tmp/browser", "http://127.0.0.1",
])
def test_only_explicit_loopback_cdp_endpoints(endpoint):
    with pytest.raises(BrowserError, match="loopback"):
        validate_endpoint(endpoint)


def test_loopback_endpoint_normalization():
    assert validate_endpoint("http://localhost:9222/") == "http://127.0.0.1:9222"
    assert validate_endpoint("http://[::1]:9222") == "http://[::1]:9222"
    assert validate_endpoint("ws://127.0.0.1:9222/devtools/browser/abc") == "ws://127.0.0.1:9222/devtools/browser/abc"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///C:/secret", "data:text/html,x", "https://user:pass@example.com", "https://", "http://example.com\n/x"])
def test_navigation_requires_http_without_credentials(url):
    with pytest.raises(BrowserError):
        validate_url(url)


def test_hub_starts_only_loopback_and_closes_idempotently():
    hub = BrowserHub(port=0)
    assert hub._server is None
    with hub:
        assert hub._server.server_address[0] == "127.0.0.1"
        assert hub.port > 0
        assert len(hub.token) >= 40
    hub.close()
    with pytest.raises(BrowserError, match="closed"):
        hub.start()


def _stop_disposable_browser(item):
    """Kill only the PID returned by our explicit test launch, never discovered PIDs."""
    try:
        parent = psutil.Process(item["pid"])
        children = parent.children(recursive=True)
        for process in children:
            try:
                process.terminate()
            except psutil.Error:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([parent, *children], timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    # The profile path is returned by launch and is always a dedicated temp dir.
    profile = Path(item["profile_dir"]).resolve()
    import tempfile
    assert profile.parent == Path(tempfile.gettempdir()).resolve()
    assert profile.name.startswith("groundwork-browser-")
    shutil.rmtree(profile, ignore_errors=True)


@pytest.fixture
def demo_server(tmp_path):
    (tmp_path / "index.html").write_text('''<!doctype html><title>Transport test</title>
      <style>body{margin:0;height:1800px}input{position:absolute;left:100px;top:100px;width:260px;height:40px}
      button{position:absolute;left:100px;top:200px;width:180px;height:40px}</style>
      <input id="text" value="original"><button onclick="document.body.dataset.clicked='yes'">Click</button>
      <script>document.querySelector('input').addEventListener('keydown',e=>{
      if(e.key==='Enter')document.body.dataset.enter='yes';});</script>''', encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_cdp_viewport_actions_stable_tabs_and_non_destructive_detach(demo_server):
    hub = BrowserHub(port=0)
    item = None
    observer = None
    try:
        try:
            item = hub.launch("chromium", demo_server, headless=True)
        except BrowserError as error:
            if "Chromium is not installed" in str(error):
                pytest.skip("Install the test browser with python -m playwright install chromium")
            raise
        found = hub.list_browsers(endpoints=[item["endpoint"]])
        assert item["id"] in {entry["id"] for entry in found}
        connection = hub.connect(item["id"])
        with pytest.raises(BrowserError, match="Select"):
            connection.describe()
        tabs = connection.list_tabs()
        selected = next(tab for tab in tabs if tab["url"] == demo_server)
        assert connection.list_tabs()[0]["id"] == tabs[0]["id"]
        connection.select_tab(selected["id"])

        # A separate CDP client inspects our disposable demo only. Production
        # transport does not inspect DOM content or use selectors for grounding.
        observer = hub._get_playwright().chromium.connect_over_cdp(item["endpoint"], no_defaults=True)
        target = next(page for page in observer.contexts[0].pages if page.url == demo_server)
        unrelated = observer.contexts[0].new_page()
        unrelated.goto(demo_server + "?unrelated")
        connection.select_tab(selected["id"])
        screenshot = connection.capture()
        with Image.open(io.BytesIO(screenshot["png"])) as png:
            assert png.size == (screenshot["width"], screenshot["height"])
        # Scrollbars must be cropped, not included and then scaled as content.
        assert screenshot["width"] / screenshot["viewport_width"] == pytest.approx(screenshot["height"] / screenshot["viewport_height"], abs=.005)
        assert screenshot["height"] < 1800
        with pytest.raises(BrowserError, match="inside"):
            connection.click(float("nan"), 20)
        with pytest.raises(BrowserError, match="inside"):
            connection.click(screenshot["viewport_width"], 20)
        connection.click(140, 120)
        connection.type_text("hello grounded browser", replace=True)
        assert target.locator("#text").input_value() == "hello grounded browser"
        connection.type_text("!", replace=False)
        connection.press("Control+A")
        connection.type_text("replacement", replace=False)
        connection.press("Enter")
        assert target.locator("#text").input_value() == "replacement"
        assert target.locator("body").get_attribute("data-enter") == "yes"
        connection.press("ControlOrMeta+A")
        connection.type_text("", replace=True)
        assert target.locator("#text").input_value() == ""
        connection.press("é")
        assert target.locator("#text").input_value() == "é"
        connection.press("F24")
        connection.click(140, 220)
        assert target.locator("body").get_attribute("data-clicked") == "yes"
        connection.scroll(0, 500)
        scrolled = connection.capture()
        assert scrolled["scroll_y"] > 0
        assert scrolled["width"] == screenshot["width"]
        assert scrolled["height"] == screenshot["height"]
        connection.navigate(demo_server + "?navigated")
        assert connection.describe()["url"].endswith("?navigated")
        connection.close()
        assert target.title() == "Transport test"
        assert not unrelated.is_closed()
        observer.close()
        observer = None
        hub.close()
        with urlopen(item["endpoint"] + "/json/list", timeout=2) as response:
            remaining = json.load(response)
        urls = {entry["url"] for entry in remaining}
        assert demo_server + "?navigated" in urls
        assert demo_server + "?unrelated" in urls
    finally:
        if observer:
            observer.close()
        hub.close()
        if item:
            _stop_disposable_browser(item)
