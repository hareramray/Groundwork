from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from grounding.browser_runtime import BrowserError, BrowserHub
import grounding.browser_runtime as browser_runtime


def call(hub, path, data=None, *, token=None, origin=None):
    headers = {"Authorization": "Bearer " + (hub.token if token is None else token)}
    if origin:
        headers["Origin"] = origin
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    with urlopen(Request(hub.bridge_url + path, data=body, headers=headers), timeout=3) as response:
        return json.load(response)


def register(hub, instance="test_browser_instance"):
    return call(hub, "/v1/register", {"instance_id": instance, "name": "Test Chrome"})["browser_id"]


def test_extension_relay_auth_origin_and_registration():
    with BrowserHub(port=0) as hub:
        with pytest.raises(HTTPError) as rejected:
            call(hub, "/v1/status", token="invalid")
        assert rejected.value.code == 401
        with pytest.raises(HTTPError) as rejected:
            call(hub, "/v1/status", origin="https://untrusted.example")
        assert rejected.value.code == 403
        assert call(hub, "/v1/status", origin="chrome-extension://" + "a" * 32)["ok"]
        with pytest.raises(HTTPError) as rejected:
            call(hub, "/v1/register", {"instance_id": "bad", "name": "test"})
        assert rejected.value.code == 400
        browser_id = register(hub)
        assert register(hub) == browser_id
        found = hub.list_browsers(endpoints=[])
        assert found == [{"id": browser_id, "name": "Test Chrome", "transport": "extension"}]
        connection = hub.connect(browser_id)
        with pytest.raises(BrowserError, match="Select"):
            connection.capture()
        call(hub, "/v1/unregister", {"browser_id": browser_id})
        assert hub.list_browsers(endpoints=[]) == []
        with pytest.raises(BrowserError, match="offline"):
            connection.list_tabs()


def test_extension_command_delivery_result_binding_and_error():
    with BrowserHub(port=0, command_timeout=2) as hub, ThreadPoolExecutor(max_workers=1) as executor:
        browser_id = register(hub)
        connection = hub.connect(browser_id)
        pending = executor.submit(connection.list_tabs)
        job = call(hub, "/v1/poll?browser_id=" + browser_id)["job"]
        assert job["command"] == "list_tabs"
        assert job["expires_at"] > time.time()
        with pytest.raises(HTTPError) as rejected:
            call(hub, "/v1/result", {"browser_id": "extension:another", "job_id": job["id"], "result": []})
        assert rejected.value.code == 404
        tabs = [{"id": "17", "title": "Local demo", "url": "http://127.0.0.1/demo"}]
        call(hub, "/v1/result", {"browser_id": browser_id, "job_id": job["id"], "result": tabs})
        assert pending.result() == tabs

        pending = executor.submit(connection.list_tabs)
        job = call(hub, "/v1/poll?browser_id=" + browser_id)["job"]
        call(hub, "/v1/result", {"browser_id": browser_id, "job_id": job["id"], "error": "Debugger was detached."})
        with pytest.raises(BrowserError, match="detached"):
            pending.result()


def test_expired_commands_are_cancelled_and_never_delivered():
    with BrowserHub(port=0, command_timeout=.05) as hub:
        browser_id = register(hub)
        with pytest.raises(BrowserError, match="outcome may be unknown"):
            hub._extension_command(browser_id, "click", {"x": 10, "y": 10})
        expired = hub._extensions[browser_id]["queue"].get_nowait()
        assert expired.cancelled
        assert expired.id not in hub._jobs
        with pytest.raises(HTTPError) as rejected:
            call(hub, "/v1/result", {"browser_id": browser_id, "job_id": expired.id, "result": {}})
        assert rejected.value.code == 404


def test_extension_disconnect_unblocks_pending_command():
    with BrowserHub(port=0, command_timeout=2) as hub, ThreadPoolExecutor(max_workers=1) as executor:
        browser_id = register(hub)
        pending = executor.submit(hub._extension_command, browser_id, "capture", {"tab_id": "17"})
        call(hub, "/v1/poll?browser_id=" + browser_id)
        call(hub, "/v1/unregister", {"browser_id": browser_id})
        with pytest.raises(BrowserError, match="disconnected"):
            pending.result(timeout=1)


def test_interrupted_wait_cancels_queued_action(monkeypatch):
    with BrowserHub(port=0) as hub:
        browser_id = register(hub)

        class InterruptedEvent:
            def wait(self, timeout):
                raise KeyboardInterrupt

        original_job = browser_runtime._Job

        def interrupted_job(*args):
            job = original_job(*args)
            job.event = InterruptedEvent()
            return job

        monkeypatch.setattr(browser_runtime, "_Job", interrupted_job)
        with pytest.raises(KeyboardInterrupt):
            hub._extension_command(browser_id, "click", {"tab_id": "17", "x": 10, "y": 10})
        abandoned = hub._extensions[browser_id]["queue"].get_nowait()
        assert abandoned.cancelled
        assert abandoned.id not in hub._jobs
