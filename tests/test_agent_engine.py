import io
import json

import pytest
from PIL import Image, ImageDraw

from grounding.agent_engine import Action, SessionEngine, Task, load_task, parse_command


def png(color="white", target_color=None):
    image = Image.new("RGB", (800, 600), color)
    if target_color:
        ImageDraw.Draw(image).rectangle((320, 240, 480, 360), fill=target_color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class Browser:
    def __init__(self):
        self.actions = []
        self.captures = 0
        self.frame = {"png": png(), "width": 800, "height": 600, "viewport_width": 400,
                      "viewport_height": 300, "url": "https://example.test", "title": "Test", "tab_id": "tab-1"}

    def capture(self):
        self.captures += 1
        return dict(self.frame)

    def click(self, x, y):
        self.actions.append(("click", x, y))

    def type_text(self, text, replace=True):
        self.actions.append(("type", text, replace))

    def press(self, key):
        self.actions.append(("press", key))

    def scroll(self, dx, dy):
        self.actions.append(("scroll", dx, dy))

    def navigate(self, url):
        self.actions.append(("navigate", url))
        self.frame["url"] = url


class Model:
    metadata = {"kind": "test fixture"}

    def __init__(self, present=True):
        self.present = present
        self.calls = []

    def predict(self, screenshot, instruction, threshold=None):
        self.calls.append((screenshot, instruction, threshold))
        return {"target_present": self.present, "presence_score": .9 if self.present else .1,
                "bbox": [.4, .4, .6, .6] if self.present else None,
                "click_point": [.5, .5] if self.present else None}


@pytest.fixture
def setup(tmp_path):
    browser, model = Browser(), Model()
    engine = SessionEngine(browser, model, tmp_path, auto=True, settle_seconds=0)
    return browser, model, engine


def test_click_uses_css_coordinates_and_observes_after(setup):
    browser, model, engine = setup
    result = engine.execute({"action": "click", "instruction": "button"})
    assert result["status"] == "executed"
    assert result["applied"] is True
    assert browser.actions == [("click", 200, 150)]
    assert browser.captures == 3
    assert len(model.calls) == 1
    assert (engine.output_dir / result["after"]["screenshot"]).is_file()


def test_find_does_not_mutate_page(setup):
    browser, model, engine = setup
    assert engine.execute({"action": "find", "instruction": "field"})["status"] == "found"
    assert browser.actions == []
    assert browser.captures == 1


def test_type_text_redacted_from_logs_and_replaces_focus(setup):
    browser, model, engine = setup
    result = engine.execute({"action": "type", "instruction": "field", "text": "MY-PRIVATE-TEXT"})
    assert result["status"] == "executed"
    assert browser.actions == [("click", 200, 150), ("type", "MY-PRIVATE-TEXT", True)]
    assert "MY-PRIVATE-TEXT" not in engine.log_path.read_text()
    assert result["action"]["text_length"] == 15


def test_type_transport_exception_does_not_leak_text(setup):
    browser, model, engine = setup
    def fail(text, replace):
        raise RuntimeError(f"Could not insert {text}")
    browser.type_text = fail
    result = engine.execute({"action": "type", "instruction": "field", "text": "TOP-SECRET"})
    assert result["status"] == "error" and result["applied"]
    assert "TOP-SECRET" not in engine.log_path.read_text()
    assert "retry_note" in result


def test_default_confirmation_cancels_without_input(tmp_path):
    browser = Browser()
    engine = SessionEngine(browser, Model(), tmp_path, settle_seconds=0)
    assert engine.execute({"action": "click", "instruction": "button"})["status"] == "cancelled"
    assert not browser.actions


@pytest.mark.parametrize("change", ["url", "tab", "viewport", "target", "whole_page", "scroll"])
def test_stale_page_or_target_after_confirmation_prevents_click(tmp_path, change):
    browser = Browser()
    def confirm(proposal):
        if change == "url":
            browser.frame["url"] += "/next"
        elif change == "tab":
            browser.frame["tab_id"] = "another-tab"
        elif change == "viewport":
            browser.frame["viewport_width"] = 401
        elif change == "target":
            browser.frame["png"] = png(target_color="black")
        elif change == "scroll":
            browser.frame["scroll_y"] = 120
        else:
            browser.frame["png"] = png("black")
        return True
    engine = SessionEngine(browser, Model(), tmp_path, confirm=confirm, settle_seconds=0)
    result = engine.execute({"action": "click", "instruction": "button"})
    assert result["status"] == "blocked"
    assert not browser.actions


def test_abstention_stops_task_before_later_actions(setup):
    browser, model, engine = setup
    model.present = False
    result = engine.run({"steps": [{"action": "click", "instruction": "absent"},
                                   {"action": "navigate", "url": "https://example.test/next"}]})
    assert result["status"] == "abstained"
    assert len(result["results"]) == 1
    assert not browser.actions


def test_run_validates_all_steps_before_actions(setup):
    browser, model, engine = setup
    with pytest.raises(ValueError):
        engine.run({"steps": [{"action": "click", "instruction": "button"}, {"action": "javascript", "code": "x"}]})
    assert not browser.actions and browser.captures == 0


def test_each_step_grounds_fresh_screenshot(setup):
    browser, model, engine = setup
    original_click = browser.click
    def click(x, y):
        original_click(x, y)
        browser.frame["png"] = png("green")
    browser.click = click
    result = engine.run({"steps": [{"action": "click", "instruction": "button"},
                                   {"action": "find", "instruction": "next"}]})
    assert result["status"] == "completed"
    assert model.calls[0][0] != model.calls[1][0]


def test_bounded_scroll_search_recaptures_and_regrounds(setup):
    browser, model, engine = setup
    model.present = False
    original_scroll = browser.scroll
    def scroll(dx, dy):
        original_scroll(dx, dy)
        browser.frame["png"] = png("green")
        model.present = True
    browser.scroll = scroll
    result = engine.execute({"action": "click", "instruction": "lower button", "search_scrolls": 2})
    assert result["status"] == "executed"
    assert browser.actions == [("scroll", 0, 210), ("click", 200, 150)]
    assert model.calls[0][0] != model.calls[1][0]
    assert engine.used_steps == 2


def test_scroll_search_stops_at_budget(setup):
    browser, model, engine = setup
    engine.max_steps = 2
    model.present = False
    result = engine.execute({"action": "click", "instruction": "absent", "search_scrolls": 10})
    assert result["status"] == "blocked"
    assert len(browser.actions) == 1
    assert browser.actions[0][0] == "scroll"


def test_task_step_budget_checked_before_execution(setup):
    browser, model, engine = setup
    engine.max_steps = 1
    result = engine.run({"steps": [{"action": "press", "key": "Enter"}, {"action": "press", "key": "Tab"}]})
    assert result["status"] == "blocked"
    assert browser.captures == 0


@pytest.mark.parametrize("bad", [[float("nan"), .5], [-.1, .5], [1, .5], [.9, .5]])
def test_invalid_model_point_cannot_click(setup, bad):
    browser, model, engine = setup
    original = model.predict
    def predict(*args, **kwargs):
        return {**original(*args, **kwargs), "click_point": bad}
    model.predict = predict
    assert engine.execute({"action": "click", "instruction": "button"})["status"] == "error"
    assert not browser.actions


def test_cancelled_approval_stops_remaining_task(tmp_path):
    browser = Browser()
    engine = SessionEngine(browser, Model(), tmp_path, confirm=lambda _: False)
    report = engine.run({"steps": [{"action": "press", "key": "Enter"}, {"action": "press", "key": "Tab"}]})
    assert report["status"] == "cancelled"
    assert not browser.actions


def test_keyboard_interrupt_records_cancellation(setup):
    browser, model, engine = setup
    def interrupt(key):
        raise KeyboardInterrupt()
    browser.press = interrupt
    result = engine.execute({"action": "press", "key": "Enter"})
    assert result["status"] == "cancelled"
    assert result["interrupted"] and result["input_attempted"] and result["outcome_unknown"]
    assert '"status": "cancelled"' in engine.log_path.read_text()


def test_transport_timeout_marks_unknown_outcome_without_retry(setup):
    browser, model, engine = setup
    def timeout(x, y):
        browser.actions.append(("click", x, y))
        raise TimeoutError("No acknowledgment")
    browser.click = timeout
    result = engine.run({"steps": [{"action": "click", "instruction": "button"}, {"action": "press", "key": "Enter"}]})
    assert result["status"] == "error"
    assert len(browser.actions) == 1
    assert result["results"][0]["outcome_unknown"]


def test_missing_model_blocks_whole_task_before_navigation(tmp_path):
    browser = Browser()
    engine = SessionEngine(browser, None, tmp_path, auto=True)
    result = engine.run({"steps": [{"action": "navigate", "url": "https://example.test/next"},
                                   {"action": "find", "instruction": "field"}]})
    assert result["status"] == "blocked" and browser.captures == 0


@pytest.mark.parametrize("command,expected", [
    ('click "Find a button"', {"action": "click", "instruction": "Find a button"}),
    ('type "hello \\"world\\"" into message field', {"action": "type", "text": 'hello "world"', "instruction": "message field"}),
    ('scroll down 450', {"action": "scroll", "dy": 450.0}),
    ('scroll left', {"action": "scroll", "dx": -600.0}),
    ('press Control+A', {"action": "press", "key": "Control+A"}),
    ('open https://example.test', {"action": "navigate", "url": "https://example.test"}),
    ('wait 0.1', {"action": "wait", "seconds": .1}),
])
def test_command_parser(command, expected):
    assert parse_command(command).model_dump(exclude_unset=True) == expected


@pytest.mark.parametrize("command", ["buy shoes", 'type no quotes into field', 'type "x"', "scroll down nan",
                                      "scroll down -1", "scroll diagonal 1", "press", "wait nan", "wait 61",
                                      "open javascript:alert(1)", 'click ""'])
def test_bad_commands_rejected(command):
    with pytest.raises(ValueError):
        parse_command(command)


@pytest.mark.parametrize("data", [
    {"action": "click", "instruction": ""}, {"action": "type", "instruction": "field"},
    {"action": "click", "instruction": "button", "selector": "#button"},
    {"action": "click", "instruction": "button", "url": "https://example.test"},
    {"action": "scroll", "dy": float("inf")}, {"action": "scroll"},
    {"action": "navigate", "url": "https://username:password@example.test"},
    {"action": "navigate", "url": "file:///private"}, {"action": "wait", "seconds": True},
    {"action": "press", "key": "Bogus+Enter"}, {"action": "press", "key": "shell code"},
    {"action": "find", "instruction": "field", "search_scrolls": 11},
])
def test_strict_task_schema(data):
    with pytest.raises(ValueError):
        Action.model_validate(data)


def test_task_utf8_bom_and_invalid_extra_field(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"steps": [{"action": "find", "instruction": "field"}]}), encoding="utf-8-sig")
    assert load_task(path).steps[0].action == "find"
    with pytest.raises(ValueError):
        Task.model_validate({"steps": [], "planner": "external"})
