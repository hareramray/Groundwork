"""CLI selection and exit behavior using fake transports, never user tabs."""
import io
import json

import pytest

from grounding import agent_cli, browser_runtime


class FakeConnection:
    def __init__(self):
        self.selected = None
        self.closed = False
        self.tabs = [
            {"id": "tab-a", "title": "First", "url": "https://first.example/"},
            {"id": "tab-b", "title": "Second", "url": "https://second.example/"},
        ]

    def list_tabs(self):
        return self.tabs

    def select_tab(self, tab_id):
        self.selected = tab_id

    def describe(self):
        if self.selected is None:
            raise ValueError("Select a tab first")
        return next(tab for tab in self.tabs if tab["id"] == self.selected)

    def close(self):
        self.closed = True


class FakeHub:
    def __init__(self):
        self.connection = FakeConnection()
        self.closed = False
        self.connected_ids = []
        self.endpoint_requests = []
        self.bridge_url = "http://127.0.0.1:12345"
        self.token = "unused-test-token"
        self.browsers = [{"id": "cdp:http://127.0.0.1:9222", "endpoint": "http://127.0.0.1:9222",
                          "name": "Test browser", "transport": "cdp"}]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        self.connection.close()

    def list_browsers(self, endpoints=None):
        self.endpoint_requests.append(endpoints)
        return self.browsers

    def connect(self, browser_id):
        self.connected_ids.append(browser_id)
        return self.connection


@pytest.fixture
def fake_hub(monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: hub)
    monkeypatch.setattr(agent_cli.sys, "stdin", io.StringIO())
    return hub


def test_paths_with_spaces_and_backslashes_are_literal():
    path = r"C:\Users\First Last\models\checkpoint.pt"
    assert agent_cli.clean_path('"' + path + '"') == path
    assert agent_cli.clean_path("'" + path + "'") == path
    assert agent_cli.clean_path(path) == path


def test_exact_tab_id_and_unique_url_selection():
    tabs = FakeConnection().tabs
    assert agent_cli.select_item(tabs, "tab-b", "tab")["id"] == "tab-b"
    assert agent_cli.select_item(tabs, "https://first.example/", "tab")["id"] == "tab-a"
    with pytest.raises(ValueError, match="No unique"):
        agent_cli.select_item(tabs + [dict(tabs[0], id="duplicate-url")], "https://first.example/", "tab")
    with pytest.raises(ValueError):
        agent_cli.select_item(tabs, "not-a-tab", "tab")


def test_numeric_tab_ids_never_alias_display_numbers():
    tabs = [{"id": "2", "title": "Displayed first"}, {"id": "1", "title": "Displayed second"}]
    assert agent_cli.select_item(tabs, "1", "tab")["title"] == "Displayed second"
    assert agent_cli.select_item(tabs, "id:1", "tab")["title"] == "Displayed second"
    assert agent_cli.select_item(tabs, "#1", "tab")["title"] == "Displayed first"
    assert agent_cli.select_item(tabs, "#2", "tab")["title"] == "Displayed second"
    with pytest.raises(ValueError):
        agent_cli.select_item(tabs, "#3", "tab")
    with pytest.raises(ValueError):
        agent_cli.select_item(FakeConnection().tabs, "1", "tab")


def test_unattended_selection_requires_disambiguation():
    tabs = FakeConnection().tabs
    with pytest.raises(ValueError, match="Multiple tabs"):
        agent_cli.choose(tabs, None, "tab", interactive=False)
    assert agent_cli.choose(tabs[:1], None, "tab", interactive=False)["id"] == "tab-a"


@pytest.mark.parametrize("arguments", [
    ["--threshold", "nan"], ["--threshold", "1.1"], ["--max-steps", "0"],
    ["--port", "65536"], ["--task", "a.json", "--command", "wait 1"],
    ["--launch", "chrome", "--endpoint", "http://127.0.0.1:9222"],
])
def test_invalid_cli_options_fail_before_browser_start(monkeypatch, arguments):
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: pytest.fail("No browser hub should start"))
    with pytest.raises(SystemExit) as error:
        agent_cli.main(arguments)
    assert error.value.code == 2


def test_list_models_is_json_without_starting_browser(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "exports"
    directory.mkdir()
    path = directory / "export.pt"
    path.write_bytes(b"discovery does not load weights")
    monkeypatch.setenv("GROUNDING_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: pytest.fail("No browser hub should start"))
    assert agent_cli.main(["--list-models"]) == 0
    assert json.loads(capsys.readouterr().out) == [str(path)]


def test_model_menu_includes_generated_step_checkpoints(tmp_path, monkeypatch):
    directory = tmp_path / "runs" / "run-1" / "checkpoints"
    directory.mkdir(parents=True)
    path = directory / "step_00000012.pt"
    path.write_bytes(b"Only discovery; model loading is separate")
    monkeypatch.setenv("GROUNDING_DATA_DIR", str(tmp_path))
    assert agent_cli.local_models() == [path]


def test_list_browsers_is_clean_json_and_detaches(fake_hub, capsys):
    assert agent_cli.main(["--list-browsers"]) == 0
    assert json.loads(capsys.readouterr().out) == fake_hub.browsers
    assert fake_hub.closed


def test_list_tabs_is_clean_json_and_detaches(fake_hub, capsys):
    assert agent_cli.main(["--endpoint", "http://127.0.0.1:9222", "--list-tabs"]) == 0
    assert json.loads(capsys.readouterr().out) == fake_hub.connection.tabs
    assert fake_hub.closed


def test_list_tabs_by_browser_id_ignores_model_and_is_clean_json(fake_hub, capsys):
    assert agent_cli.main(["--browser-id", fake_hub.browsers[0]["id"], "--model", "unused.pt", "--list-tabs"]) == 0
    assert json.loads(capsys.readouterr().out) == fake_hub.connection.tabs


def test_normalized_localhost_endpoint_matches_discovered_browser(fake_hub, capsys):
    assert agent_cli.main(["--endpoint", "http://localhost:9222/", "--list-tabs"]) == 0
    assert fake_hub.connected_ids == [fake_hub.browsers[0]["id"]]
    assert fake_hub.closed


def test_unavailable_endpoint_keeps_interactive_bridge_open_for_extension(fake_hub, monkeypatch, capsys):
    # A different discovered browser must not be chosen as an automatic fallback.
    fake_hub.browsers = [{"id": "cdp:http://127.0.0.1:9223", "endpoint": "http://127.0.0.1:9223",
                          "name": "Other browser", "transport": "cdp"}]
    monkeypatch.setattr(agent_cli.sys.stdin, "isatty", lambda: True)
    loaded_models = []
    monkeypatch.setattr(agent_cli.Console, "load_model", lambda self, path=None: loaded_models.append(path))
    prompts = []
    commands = iter(["browsers", "connect extension-test", "use tab-b", "quit"])

    def respond(prompt):
        assert prompt == "groundwork> "
        assert not fake_hub.closed
        if not prompts:
            assert fake_hub.connected_ids == []
            fake_hub.browsers.append({"id": "extension-test", "name": "My browser", "transport": "extension"})
        prompts.append(prompt)
        return next(commands)

    monkeypatch.setattr("builtins.input", respond)
    assert agent_cli.main(["--model", "test.pt", "--endpoint", "http://127.0.0.1:9222"]) == 0
    output = capsys.readouterr()
    assert "No browser responded at --endpoint http://127.0.0.1:9222" in output.err
    assert "does not launch" in output.err
    assert "--launch edge" in output.err
    assert "bridge are still running" in output.out
    assert fake_hub.connected_ids == ["extension-test"]
    assert fake_hub.connection.selected == "tab-b"
    assert [str(path) for path in loaded_models] == ["test.pt"]
    assert len(prompts) == 4
    assert fake_hub.closed


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("arguments", [["--list-tabs"], ["--command", "wait 0", "--auto"], ["--task"]])
def test_unavailable_endpoint_still_fails_for_batch_requests(fake_hub, monkeypatch, capsys, tmp_path, terminal, arguments):
    fake_hub.browsers = []
    monkeypatch.setattr(agent_cli.sys.stdin, "isatty", lambda: terminal)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("Batch requests must not enter the prompt"))
    monkeypatch.setattr(agent_cli.Console, "get_engine", lambda self: pytest.fail("No task may run without the requested browser"))
    if arguments == ["--task"]:
        task = tmp_path / "task.json"
        task.write_text(json.dumps({"steps": [{"action": "wait", "seconds": 0}]}))
        arguments = ["--task", str(task)]
    assert agent_cli.main(["--endpoint", "http://127.0.0.1:9222", *arguments]) == 2
    output = capsys.readouterr()
    assert "No browser responded at --endpoint http://127.0.0.1:9222" in output.err
    assert "--launch edge" in output.err
    assert "still running" not in output.out
    assert fake_hub.connected_ids == []
    assert fake_hub.closed


def test_noninteractive_repl_is_rejected_before_browser_start(monkeypatch, capsys):
    monkeypatch.setattr(agent_cli.sys, "stdin", io.StringIO())
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: pytest.fail("No browser hub should start"))
    assert agent_cli.main([]) == 2
    assert "needs a terminal" in capsys.readouterr().err


def test_invalid_task_does_not_start_browser(tmp_path, monkeypatch, capsys):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"steps": [{"action": "javascript", "script": "anything"}]}))
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: pytest.fail("No browser hub should start"))
    assert agent_cli.main(["--task", str(path), "--auto"]) == 2
    assert "Error:" in capsys.readouterr().err


def test_missing_model_for_grounded_task_fails_before_browser_start(tmp_path, monkeypatch, capsys):
    started = []
    monkeypatch.setattr(browser_runtime, "BrowserHub", lambda **kwargs: started.append(True) or FakeHub())
    code = agent_cli.main(["--command", "open https://example.com", "--command", "click button",
                           "--endpoint", "http://127.0.0.1:9222", "--tab", "tab-a", "--auto",
                           "--output-dir", str(tmp_path)])
    assert code == 2
    assert started == []
    assert "model" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("status,expected_code", [("completed", 0), ("blocked", 3), ("abstained", 3),
                                                  ("cancelled", 3), ("error", 3)])
def test_script_report_status_sets_exit_code_and_detaches(fake_hub, monkeypatch, status, expected_code):
    class Engine:
        def run(self, task):
            assert len(task.steps) == 1
            return {"status": status, "results": []}
    monkeypatch.setattr(agent_cli.Console, "get_engine", lambda self: Engine())
    code = agent_cli.main(["--endpoint", "http://127.0.0.1:9222", "--tab", "tab-b",
                           "--command", "wait 0", "--auto"])
    assert code == expected_code
    assert fake_hub.connection.selected == "tab-b"
    assert fake_hub.closed


def test_interrupt_detaches_and_returns_130(fake_hub, monkeypatch):
    class Engine:
        def run(self, task):
            raise KeyboardInterrupt
    monkeypatch.setattr(agent_cli.Console, "get_engine", lambda self: Engine())
    assert agent_cli.main(["--endpoint", "http://127.0.0.1:9222", "--tab", "tab-a",
                           "--command", "wait 0"]) == 130
    assert fake_hub.closed


def test_explicit_tab_is_required_for_ambiguous_unattended_browser(fake_hub, capsys):
    assert agent_cli.main(["--endpoint", "http://127.0.0.1:9222", "--command", "wait 0", "--auto"]) == 2
    assert fake_hub.connection.selected is None
    assert "Multiple tabs" in capsys.readouterr().err
    assert fake_hub.closed


@pytest.mark.parametrize("command,instruction", [
    ('type "apple" in the search textbox', 'the search textbox'),
    ('type "apple" in the textbox', 'the textbox'),
    ('type "apple" in "find the search field and click on that"', 'find the search field and click on that'),
    ('type "apple" in the "find the search field and click on that"', 'find the search field and click on that'),
])
def test_repl_dispatches_type_in_forms_to_engine(tmp_path, monkeypatch, capsys, command, instruction):
    calls = []

    class Engine:
        def execute(self, action):
            calls.append(action.model_dump(exclude_unset=True))
            return {"status": "executed", "action": action.log_data()}

    console = agent_cli.Console(agent_cli.parser().parse_args(["--output-dir", str(tmp_path)]), FakeHub())
    monkeypatch.setattr(console, "get_engine", lambda: Engine())
    assert console.command(command) is True
    assert calls == [{"action": "type", "text": "apple", "instruction": instruction}]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "executed"
    assert result["action"]["instruction"] == instruction
    assert result["action"]["text_length"] == 5
    assert "text" not in result["action"]
