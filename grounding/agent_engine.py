"""Local screenshot-grounded commands. Intent comes only from the user's task file."""
from __future__ import annotations

import io
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

from PIL import Image, ImageChops, ImageStat
from pydantic import BaseModel, ConfigDict, Field, model_validator


TARGET_ACTIONS = {"find", "assert_visible", "click", "type"}
SUCCESS = {"executed", "found"}


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    action: Literal["find", "assert_visible", "click", "type", "press", "scroll", "navigate", "wait"]
    instruction: str | None = Field(default=None, max_length=4096)
    text: str | None = Field(default=None, max_length=100_000)
    replace: bool = True
    key: str | None = None
    dx: float = Field(default=0, ge=-10000, le=10000)
    dy: float = Field(default=0, ge=-10000, le=10000)
    url: str | None = Field(default=None, max_length=8192)
    seconds: float | None = Field(default=None, ge=0, le=60)
    search_scrolls: int = Field(default=0, ge=0, le=10)

    @model_validator(mode="after")
    def check_action(self):
        allowed = {"action"}
        if self.action in TARGET_ACTIONS:
            allowed |= {"instruction", "search_scrolls"}
            if not self.instruction or not self.instruction.strip():
                raise ValueError(f"{self.action} needs a nonempty instruction")
        if self.action == "type":
            allowed |= {"text", "replace"}
            if self.text is None:
                raise ValueError("type needs text")
        if self.action == "press":
            allowed.add("key")
            validate_key(self.key)
        if self.action == "scroll":
            allowed |= {"dx", "dy"}
            if not self.dx and not self.dy:
                raise ValueError("scroll needs a nonzero dx or dy")
        if self.action == "navigate":
            allowed.add("url")
            validate_url(self.url)
        if self.action == "wait":
            allowed.add("seconds")
            if self.seconds is None:
                raise ValueError("wait needs seconds (0 to 60)")
        extra = self.model_fields_set - allowed
        if extra:
            raise ValueError(f"Fields do not apply to {self.action}: {', '.join(sorted(extra))}")
        return self

    def log_data(self) -> dict:
        data = self.model_dump(exclude_unset=True)
        if "text" in data:
            data["text_length"] = len(data.pop("text"))
        return data


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(default="Local browser task", min_length=1, max_length=256)
    steps: list[Action] = Field(min_length=1, max_length=1000)


def validate_url(url: str | None) -> str:
    try:
        parsed = urlsplit(url or "")
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
        parsed.port  # Reject malformed ports before any browser operation.
    except ValueError:
        valid = False
    if not valid or any(ord(char) < 32 for char in (url or "")):
        raise ValueError("Use a complete http:// or https:// URL without embedded credentials")
    return url


def validate_key(key: str | None) -> str:
    if not key or len(key) > 80:
        raise ValueError("press needs a key such as Enter, Tab, Escape, or Control+A")
    parts = key.split("+")
    modifiers = {"Control", "Alt", "Shift", "Meta", "ControlOrMeta"}
    named = {"Enter", "Tab", "Escape", "Backspace", "Delete", "Space", "ArrowUp", "ArrowDown",
             "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", "Insert"}
    final = parts[-1]
    if any(part not in modifiers for part in parts[:-1]) or not (
        (len(final) == 1 and final.isprintable()) or final in named or re.fullmatch(r"F([1-9]|1[0-9]|2[0-4])", final)
    ):
        raise ValueError("Unsupported key; use Enter, Tab, Escape, ArrowDown, Control+A, etc.")
    return key


def _unquote(text: str) -> str:
    text = text.strip()
    if text.startswith('"'):
        value = json.loads(text)
        if not isinstance(value, str):
            raise ValueError("Expected a quoted string")
        return value
    return text


def parse_command(command: str) -> Action:
    """Parse explicit REPL commands, never a natural-language plan or page content."""
    parts = command.strip().split(maxsplit=1)
    verb, value = parts[0].lower() if parts else "", parts[1].strip() if len(parts) > 1 else ""
    if verb in {"find", "click", "assert_visible"}:
        return Action(action=verb, instruction=_unquote(value))
    if verb == "type":
        usage = 'Use: type "text to enter" in|into instruction describing the field'
        try:
            text, end = json.JSONDecoder().raw_decode(value)
            target = value[end:].split(maxsplit=1)
            if not isinstance(text, str) or len(target) != 2 or target[0].lower() not in {"in", "into"}:
                raise ValueError(usage)
            instruction = target[1].strip()
            # Accept 'in the "search field"' without adding wrapper words/quotes
            # to the model prompt; leave ordinary unquoted descriptions intact.
            article = re.match(r'^the\s+(?=")', instruction, flags=re.IGNORECASE)
            if article:
                quoted = instruction[article.end():]
                _, quoted_end = json.JSONDecoder().raw_decode(quoted)
                if not quoted[quoted_end:].strip():
                    instruction = quoted
            instruction = _unquote(instruction)
            if not instruction.strip():
                raise ValueError(usage)
        except ValueError as exc:
            raise ValueError(usage) from exc
        return Action(action="type", text=text, instruction=instruction)
    if verb == "press":
        return Action(action="press", key=value)
    if verb in {"open", "navigate"}:
        return Action(action="navigate", url=_unquote(value))
    if verb == "wait":
        return Action(action="wait", seconds=float(value))
    if verb == "scroll":
        parts = value.split()
        if len(parts) not in {1, 2} or parts[0] not in {"up", "down", "left", "right"}:
            raise ValueError("Use: scroll up|down|left|right [pixels]")
        pixels = float(parts[1]) if len(parts) == 2 else 600.0
        if not math.isfinite(pixels) or pixels <= 0:
            raise ValueError("Scroll pixels must be positive and finite")
        return Action(action="scroll", **{("dx" if parts[0] in {"left", "right"} else "dy"):
                      pixels * (-1 if parts[0] in {"up", "left"} else 1)})
    raise ValueError("Unknown action. Use help for commands; arbitrary goals need explicit steps.")


def load_task(path: str | Path) -> Task:
    source = Path(path).expanduser()
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Task files must be smaller than 2 MiB")
    return Task.model_validate(json.loads(source.read_text(encoding="utf-8-sig")))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionEngine:
    def __init__(self, connection, model=None, output_dir: str | Path | None = None, *, auto: bool = False,
                 confirm: Callable[[dict], bool] | None = None, threshold: float = 0.5,
                 max_steps: int = 50, settle_seconds: float = 0.25):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between zero and one")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer from 1 to 1000")
        if not math.isfinite(settle_seconds) or not 0 <= settle_seconds <= 10:
            raise ValueError("settle_seconds must be from 0 to 10")
        self.connection, self.model = connection, model
        self.auto, self.confirm, self.threshold = auto, confirm, threshold
        self.max_steps, self.settle_seconds, self.used_steps = max_steps, settle_seconds, 0
        self.sequence = 0
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.output_dir = Path(output_dir) if output_dir else Path("data/agent_sessions") / stamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "actions.jsonl"
        self._log({"event": "session", "auto": auto, "threshold": threshold, "max_steps": max_steps,
                   "model": getattr(model, "metadata", None)})

    def _log(self, value: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": _now(), **value}, ensure_ascii=False, allow_nan=False) + "\n")

    def _capture(self, suffix: str) -> dict:
        shot = self.connection.capture()
        for key in ("width", "height", "viewport_width", "viewport_height"):
            value = shot.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Browser returned an invalid {key}")
        if not isinstance(shot.get("png"), bytes):
            raise ValueError("Browser did not return PNG screenshot bytes")
        # Each session/step gets distinct names even if the caller reuses output_dir.
        name = f"{self.sequence:04d}-{suffix}-{uuid.uuid4().hex[:6]}.png"
        (self.output_dir / name).write_bytes(shot["png"])
        return {**shot, "screenshot": name}

    @staticmethod
    def _observation(shot: dict) -> dict:
        return {key: value for key, value in shot.items() if key != "png"}

    @staticmethod
    def _point(prediction: dict, shot: dict) -> tuple[float, float]:
        box, point = prediction.get("bbox"), prediction.get("click_point")
        if not isinstance(box, (tuple, list)) or len(box) != 4 or not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("Model returned no valid target box/click point")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [*box, *point]):
            raise ValueError("Model returned nonfinite target coordinates")
        x1, y1, x2, y2 = box
        x, y = point
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and x1 <= x <= x2 and y1 <= y <= y2 and x < 1 and y < 1):
            raise ValueError("Model target is outside the visible viewport")
        # The screenshot can be physical pixels (DPR=2). Input is always CSS pixels.
        return x * shot["viewport_width"], y * shot["viewport_height"]

    @staticmethod
    def _stale(before: dict, current: dict, prediction: dict | None) -> bool:
        if any(before.get(key) != current.get(key) for key in
               ("tab_id", "url", "width", "height", "viewport_width", "viewport_height", "scroll_x", "scroll_y")):
            return True
        if prediction is None:
            return False
        with Image.open(io.BytesIO(before["png"])) as first, Image.open(io.BytesIO(current["png"])) as second:
            first, second = first.convert("RGB"), second.convert("RGB")
            if first.size != second.size:
                return True
            diff = ImageChops.difference(first, second)
            # Ignore tiny antialiasing/caret changes, stop on meaningful page/target changes.
            global_change = sum(ImageStat.Stat(diff).mean) / (3 * 255)
            box = prediction["bbox"]
            region = (math.floor(box[0] * first.width), math.floor(box[1] * first.height),
                      max(math.floor(box[0] * first.width) + 1, math.ceil(box[2] * first.width)),
                      max(math.floor(box[1] * first.height) + 1, math.ceil(box[3] * first.height)))
            target_change = sum(ImageStat.Stat(diff.crop(region)).mean) / (3 * 255)
            return global_change > 0.025 or target_change > 0.08

    def _approve(self, action: dict, shot: dict, prediction: dict | None = None, point=None) -> bool:
        proposal = {"action": action, "tab": self._observation(shot), "prediction": prediction,
                    "click_css": list(point) if point else None}
        self._log({"event": "proposal", **proposal})
        return self.auto or bool(self.confirm and self.confirm(proposal))

    def execute(self, action: Action | dict) -> dict:
        action = action if isinstance(action, Action) else Action.model_validate(action)
        self.sequence += 1
        result = {"step": self.sequence, "action": action.log_data(), "status": "error", "applied": False,
                  "input_attempted": False, "outcome_unknown": False}
        input_in_flight = False
        try:
            if self.used_steps >= self.max_steps:
                result.update(status="blocked", message="Session step budget exhausted")
                return result
            self.used_steps += 1
            before = self._capture("before")
            result["before"] = self._observation(before)
            prediction = None
            point = None
            if action.action in TARGET_ACTIONS:
                if self.model is None:
                    raise ValueError("Load a generated .pt model first with model <path>")
                for attempt in range(action.search_scrolls + 1):
                    prediction = self.model.predict(before["png"], action.instruction, threshold=self.threshold)
                    json.dumps(prediction, allow_nan=False)  # Invalid model values must not corrupt the action log.
                    result["prediction"] = prediction
                    if prediction.get("target_present"):
                        point = self._point(prediction, before)
                        break
                    if attempt == action.search_scrolls:
                        result.update(status="abstained", message="No confident target. No target action was performed.")
                        return result
                    if self.used_steps >= self.max_steps:
                        result.update(status="blocked", message="Step budget exhausted during scroll search")
                        return result
                    search = {"action": "scroll", "dy": round(before["viewport_height"] * 0.7), "search_attempt": attempt + 1}
                    if not self._approve(search, before):
                        result.update(status="cancelled", message="Scroll search cancelled")
                        return result
                    current = self._capture(f"search-{attempt + 1}-guard")
                    if self._stale(before, current, None):
                        result.update(status="blocked", message="Page or viewport changed. Run the step again.")
                        return result
                    self.used_steps += 1
                    result["input_attempted"] = input_in_flight = True
                    self.connection.scroll(0, search["dy"])
                    input_in_flight = False
                    result["applied"] = True
                    time.sleep(self.settle_seconds)
                    before = self._capture(f"search-{attempt + 1}")
                    self._log({"event": "search_scroll", "step": self.sequence, "action": search,
                               "observation": self._observation(before), "prediction": prediction})
                result["target_observation"] = self._observation(before)
                result["click_css"] = list(point)
                if action.action in {"find", "assert_visible"}:
                    result.update(status="found", message="Model predicted the target in the visible viewport")
                    return result
            if action.action != "wait":
                if not self._approve(action.log_data(), before, prediction, point):
                    result.update(status="cancelled", message="Action cancelled")
                    return result
                current = self._capture("guard")
                if self._stale(before, current, prediction):
                    result.update(status="blocked", message="Page or target changed after observation. Run the step again.")
                    return result
            if action.action == "click":
                result["input_attempted"] = input_in_flight = True
                self.connection.click(*point)
                input_in_flight = False
                result["applied"] = True
            elif action.action == "type":
                result["input_attempted"] = input_in_flight = True
                self.connection.click(*point)
                input_in_flight = False
                result["applied"] = True
                input_in_flight = True
                self.connection.type_text(action.text, replace=action.replace)
                input_in_flight = False
            elif action.action == "press":
                result["input_attempted"] = input_in_flight = True
                self.connection.press(action.key)
                input_in_flight = False
                result["applied"] = True
            elif action.action == "scroll":
                result["input_attempted"] = input_in_flight = True
                self.connection.scroll(action.dx, action.dy)
                input_in_flight = False
                result["applied"] = True
            elif action.action == "navigate":
                result["input_attempted"] = input_in_flight = True
                self.connection.navigate(action.url)
                input_in_flight = False
                result["applied"] = True
            elif action.action == "wait":
                time.sleep(action.seconds)
            time.sleep(self.settle_seconds)
            result["after"] = self._observation(self._capture("after"))
            result.update(status="executed", message="Step executed; inspect the resulting page or add assert_visible")
            return result
        except KeyboardInterrupt:
            result.update(status="cancelled", interrupted=True, outcome_unknown=input_in_flight,
                          message="Interrupted. Inspect the page before retrying any attempted action.")
            return result
        except Exception as exc:
            # Transport errors may echo type text, so keep only its exception type for type actions.
            detail = type(exc).__name__ if action.action == "type" else str(exc)
            result.update(status="error", message=detail, outcome_unknown=input_in_flight,
                          retry_note="Inspect the page before retrying; a browser error may occur after input was applied.")
            return result
        finally:
            self._log({"event": "step", **result})

    def run(self, task: Task | dict) -> dict:
        task = task if isinstance(task, Task) else Task.model_validate(task)
        remaining = self.max_steps - self.used_steps
        if self.model is None and any(step.action in TARGET_ACTIONS for step in task.steps):
            report = {"name": task.name, "status": "blocked", "results": [],
                      "message": "Load a generated .pt model before running this task"}
        elif len(task.steps) > remaining:
            report = {"name": task.name, "status": "blocked", "results": [],
                      "message": f"Task has {len(task.steps)} steps; session budget has {remaining} remaining"}
        else:
            report = {"name": task.name, "status": "completed", "results": []}
            for action in task.steps:
                result = self.execute(action)
                report["results"].append(result)
                if result["status"] not in SUCCESS:
                    report["status"] = result["status"]
                    break
        report["output_dir"] = str(self.output_dir.resolve())
        report["meaning"] = "Completed means the explicit steps ran; it does not prove the website goal succeeded."
        report_path = self.output_dir / f"task-{uuid.uuid4().hex[:8]}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        self._log({"event": "task", "name": task.name, "status": report["status"], "report": report_path.name})
        return report
