"""Interactive local browser agent: python -m grounding.agent_cli --help."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
HELP = '''Browser selection:
  browsers                          List connected browsers
  connect [#number or browser ID]    Select a browser
  tabs                              List open websites
  use [#number, tab ID, or URL]       Select the website to control
  launch edge|chrome|chromium [URL]   Open a separate browser profile
  model [path.pt] / models           Load or list generated models
  status / screenshot               Show selection / save viewport

Local actions (model required for find/click/type/assert_visible):
  find instruction                  Predict a target without clicking
  click instruction                 Predict then click
  type "text" in|into instruction    Predict field, click, replace text
  press Enter                       Send key (also Tab, Control+A, etc.)
  scroll up|down|left|right [pixels]  Scroll visible page
  open https://example.com           Navigate the selected tab
  wait 1                            Wait up to 60 seconds
  assert_visible instruction        Stop task if model does not find target
  run path/to/task.json              Run explicit JSON steps
  help / quit                       Help / detach, keeping browser open

The .pt model locates targets; explicit commands/task files supply the plan.
Actions ask for confirmation unless started with --auto. Ctrl+C cancels.
Screenshots and action logs stay in the local session directory.
'''


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local screenshot-grounded browser CLI using your generated .pt model.",
                                     epilog="Start without --task/--command for interactive selection and commands.")
    result.add_argument("--model", type=Path, help="Generated inference export or training checkpoint .pt")
    result.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--threshold", type=float, default=0.5, help="Presence threshold, from 0 to 1")
    result.add_argument("--trust-checkpoint", action="store_true", help="Allow unsafe pickle fallback ONLY for a checkpoint you trust")
    result.add_argument("--browser-id", help="ID printed by browsers")
    result.add_argument("--endpoint", help="Attach to an already running loopback CDP browser (does not launch one), e.g. http://127.0.0.1:9222")
    result.add_argument("--tab", help="Tab ID, #list-number (e.g. '#1'), or an exact unique URL")
    result.add_argument("--launch", choices=["edge", "chrome", "chromium"], help="Launch a separate browser profile")
    result.add_argument("--url", default="about:blank", help="Initial HTTP(S) URL for --launch")
    result.add_argument("--port", type=int, default=8766, help="Loopback extension bridge port")
    result.add_argument("--task", type=Path, help="Run a JSON task and exit")
    result.add_argument("--command", action="append", default=[], help="Run an explicit action and exit; repeat for multiple steps")
    result.add_argument("--auto", action="store_true", help="Execute your supplied sequence without per-action confirmation")
    result.add_argument("--max-steps", type=int, default=50, help="Session budget, including scroll search (1 to 1000)")
    result.add_argument("--output-dir", type=Path, help="Local action reports and screenshots directory")
    result.add_argument("--list-browsers", action="store_true", help="Print connected browsers as JSON and exit")
    result.add_argument("--list-tabs", action="store_true", help="Print selected browser tabs as JSON and exit")
    result.add_argument("--list-models", action="store_true", help="Print local generated model paths and exit")
    return result


def local_models() -> list[Path]:
    data = Path(os.environ.get("GROUNDING_DATA_DIR", ROOT / "data"))
    candidates = list((data / "exports").glob("*.pt")) + list((data / "runs").glob("*/checkpoints/*.pt"))
    # Generated runs store step_NNNNNNNN.pt files; these are selectable full checkpoints.
    candidates = [path for path in candidates if path.is_file()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[:100]


def clean_path(text: str) -> str:
    # A literal Windows path must retain its backslashes (do not use shlex).
    text = text.strip()
    return text[1:-1] if len(text) > 1 and text[0] == text[-1] and text[0] in {'"', "'"} else text


def select_item(items: list[dict], selector: str, label: str) -> dict:
    # Menu numbers use #N. Raw IDs can be numeric in a browser extension.
    if selector.startswith("#") and selector[1:].isdecimal() and 1 <= int(selector[1:]) <= len(items):
        return items[int(selector[1:]) - 1]
    identifier = selector.removeprefix("id:")
    exact = [item for item in items if str(item["id"]) == identifier]
    if len(exact) == 1:
        return exact[0]
    urls = [item for item in items if item.get("url") == selector]
    if len(urls) == 1:
        return urls[0]
    raise ValueError(f"No unique {label} matches {selector!r}; list available choices again")


def choose(items: list[dict], selector: str | None, label: str, interactive: bool) -> dict | None:
    if not items:
        raise ValueError(f"No {label}s are available")
    if selector:
        return select_item(items, selector, label)
    if not interactive:
        if len(items) == 1:
            return items[0]
        raise ValueError(f"Multiple {label}s found. Supply --{'browser-id' if label == 'browser' else 'tab'} to choose one")
    value = input(f"Select {label} #number or ID (Enter to skip): ").strip()
    return select_item(items, value, label) if value else None


class Console:
    def __init__(self, args, hub):
        self.args, self.hub = args, hub
        self.connection = None
        self.browser = None
        self.model = None
        self.engine = None
        self.session_dir = args.output_dir
        self.interactive = sys.stdin.isatty()

    def browsers(self) -> list[dict]:
        items = self.hub.list_browsers(endpoints=[self.args.endpoint] if self.args.endpoint else None)
        stream = sys.stderr if self.args.list_tabs else sys.stdout
        for index, item in enumerate(items, 1):
            print(f"  #{index}. {item['name']} [{item['transport']}]  ID={item['id']}", file=stream)
        if not items:
            print("No connected browsers. Connect the extension using the address/token above, then run browsers.", file=stream)
            print("You can also use launch edge or launch chrome to open a separate local profile.", file=stream)
        return items

    def connect(self, selector=None, items=None):
        items = self.browsers() if items is None else items
        if not items:
            return
        selected = choose(items, selector, "browser", self.interactive)
        if selected is None:
            return
        connection = self.hub.connect(selected["id"])
        if self.connection:
            self.connection.close()
        self.connection, self.browser = connection, selected
        if self.engine:
            self.engine.connection = connection
        print(f"Connected: {selected['name']}. Select a tab with use.", file=sys.stderr if self.args.list_tabs else sys.stdout)

    def tabs(self) -> list[dict]:
        if not self.connection:
            raise ValueError("Select a browser first with connect")
        tabs = self.connection.list_tabs()
        for index, tab in enumerate(tabs, 1):
            print(f"  #{index}. {tab.get('title') or '(untitled)'}\n     {tab.get('url', '')}\n     ID={tab['id']}")
        return tabs

    def use(self, selector=None):
        tabs = self.tabs()
        chosen = choose(tabs, selector, "tab", self.interactive)
        if chosen is not None:
            self.connection.select_tab(chosen["id"])
            print(f"Selected: {chosen['title']} — {chosen['url']}")

    def load_model(self, path=None):
        if path is None:
            items = local_models()
            for index, item in enumerate(items, 1):
                print(f"  {index}. {item}")
            if not self.interactive:
                raise ValueError("Supply --model with a generated .pt path")
            choice = input("Select model number or enter .pt path (Enter to skip): ").strip()
            if not choice:
                return
            path = items[int(choice) - 1] if choice.isdecimal() and 1 <= int(choice) <= len(items) else clean_path(choice)
        from .agent_model import FileGrounder
        model = FileGrounder(Path(path).expanduser(), device=self.args.device, threshold=self.args.threshold,
                             trust_checkpoint=self.args.trust_checkpoint)
        self.model = model
        if self.engine:
            self.engine.model = model
            self.engine._log({"event": "model_changed", "model": model.metadata})
        print(f"Loaded {Path(path).name} on {model.metadata.get('device', self.args.device)}")

    def confirm(self, proposal):
        action = proposal["action"]
        print(f"Proposed: {json.dumps(action, ensure_ascii=False)}")
        print(f"Tab: {proposal['tab'].get('title', '')} — {proposal['tab'].get('url', '')}")
        if proposal.get("prediction"):
            print(f"Presence score: {proposal['prediction']['presence_score']:.4f}; CSS point: {proposal['click_css']}")
        print(f"Screenshot: {self.get_engine().output_dir / proposal['tab']['screenshot']}")
        if not self.interactive:
            print("No interactive terminal. Use --auto to execute the supplied steps unattended.")
            return False
        return input("Execute this action? [y/N] ").strip().lower() in {"y", "yes"}

    def get_engine(self):
        if not self.connection:
            raise ValueError("Select a browser and tab first using connect and use")
        self.connection.describe()  # Gives an actionable missing/closed-tab error before execution.
        if self.engine is None:
            from .agent_engine import SessionEngine
            self.engine = SessionEngine(self.connection, self.model, output_dir=self.session_dir,
                                        auto=self.args.auto, confirm=self.confirm, threshold=self.args.threshold,
                                        max_steps=self.args.max_steps)
            print(f"Local session: {self.engine.output_dir.resolve()}")
        return self.engine

    def show_result(self, result):
        print(json.dumps(result, indent=2, ensure_ascii=False))

    def command(self, text: str) -> bool:
        from .agent_engine import load_task, parse_command
        verb, _, value = text.strip().partition(" ")
        verb, value = verb.lower(), value.strip()
        if not verb:
            return True
        if verb in {"quit", "exit"}:
            return False
        if verb == "help":
            print(HELP)
        elif verb == "browsers":
            self.browsers()
        elif verb == "connect":
            self.connect(value or None)
        elif verb == "tabs":
            self.tabs()
        elif verb == "use":
            self.use(clean_path(value) or None)
        elif verb == "models":
            for index, item in enumerate(local_models(), 1):
                print(f"  {index}. {item}")
        elif verb == "model":
            self.load_model(clean_path(value) or None)
        elif verb == "status":
            self.show_result({"browser": self.browser, "tab": self.connection.describe() if self.connection else None,
                              "model": getattr(self.model, "metadata", None), "auto": self.args.auto,
                              "steps_used": self.engine.used_steps if self.engine else 0,
                              "max_steps": self.args.max_steps})
        elif verb == "launch":
            browser, _, url = value.partition(" ")
            if browser not in {"edge", "chrome", "chromium"}:
                raise ValueError("Use: launch edge|chrome|chromium [http(s) URL]")
            item = self.hub.launch(browser=browser, url=url.strip() or "about:blank")
            self.connect(item["id"], items=[item])
            self.use()
        elif verb == "screenshot":
            engine = self.get_engine()
            shot = self.connection.capture()
            target = engine.output_dir / f"manual-{uuid.uuid4().hex[:10]}.png"
            target.write_bytes(shot["png"])
            print(f"Saved: {target.resolve()}")
        elif verb == "run":
            task = load_task(clean_path(value))
            if self.model is None and any(step.action in {"find", "assert_visible", "click", "type"} for step in task.steps):
                raise ValueError("Load a .pt model before running a task with target actions")
            self.show_result(self.get_engine().run(task))
        else:
            self.show_result(self.get_engine().execute(parse_command(text)))
        return True


def main(argv=None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.task and args.command:
        arg_parser.error("Use either --task or --command")
    if args.launch and (args.browser_id or args.endpoint):
        arg_parser.error("Use --launch or an existing --endpoint/--browser-id")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        arg_parser.error("--threshold must be between 0 and 1")
    if not 1 <= args.max_steps <= 1000:
        arg_parser.error("--max-steps must be from 1 to 1000")
    if not 0 <= args.port <= 65535:
        arg_parser.error("--port must be from 0 to 65535")
    if args.list_models:
        print(json.dumps([str(path) for path in local_models()], indent=2))
        return 0
    try:
        from .agent_engine import Task, load_task, parse_command
        task = load_task(args.task) if args.task else (
            Task(name="CLI commands", steps=[parse_command(command) for command in args.command]) if args.command else None)
        if task and len(task.steps) > args.max_steps:
            raise ValueError("Task exceeds --max-steps; no browser actions were performed")
        if task and args.model is None and any(step.action in {"find", "assert_visible", "click", "type"} for step in task.steps):
            raise ValueError("Supply --model for tasks containing find, assert_visible, click, or type; no actions were performed")
        if not sys.stdin.isatty() and not (task or args.list_browsers or args.list_tabs):
            raise ValueError("Interactive mode needs a terminal; use --task, --command, or a --list option")
        from .browser_runtime import BrowserHub, validate_endpoint
        if args.endpoint:
            args.endpoint = validate_endpoint(args.endpoint)
        with BrowserHub(port=args.port) as hub:
            console = Console(args, hub)
            if args.list_browsers:
                print(json.dumps(hub.list_browsers(endpoints=[args.endpoint] if args.endpoint else None), indent=2))
                return 0
            if not task and not args.list_tabs:
                print("Groundwork local browser agent — explicit commands and task files")
                print(f"Extension bridge: {hub.bridge_url}\nSession token: {hub.token}")
                print(f"Load unpacked extension from: {ROOT / 'browser-extension'}")
                print("Type help for commands. Inference is local; screenshots/logs are saved locally.")
            if args.model and not args.list_tabs:
                console.load_model(args.model)
            elif not task and not args.list_tabs:
                console.load_model()
            if args.launch:
                item = hub.launch(browser=args.launch, url=args.url)
                console.connect(item["id"], items=[item])
            elif args.endpoint:
                items = hub.list_browsers(endpoints=[args.endpoint])
                wanted = args.endpoint.rstrip("/")
                selected = [item for item in items if item.get("endpoint", "").rstrip("/") == wanted]
                if not selected:
                    message = (
                        f"No browser responded at --endpoint {args.endpoint}. "
                        "--endpoint only attaches to a browser with remote debugging already enabled; it does not launch one. "
                        "Replace --endpoint with --launch edge or --launch chrome to open a separate profile. "
                        "For existing tabs, start without --endpoint and connect the extension."
                    )
                    if task or args.list_tabs:
                        raise ValueError(message)
                    print(f"Warning: {message}", file=sys.stderr)
                    print("The CLI and extension bridge are still running. Run launch edge or launch chrome, "
                          "or connect the extension using the address/token above, then run browsers, connect, and use. "
                          "If you enable CDP at the requested endpoint, run browsers, connect, and use to retry.")
                else:
                    console.connect(args.browser_id or selected[0]["id"], items=items)
            elif args.browser_id:
                console.connect(args.browser_id)
            elif not task and not args.list_tabs:
                console.connect()
            else:
                raise ValueError("Supply --endpoint, --browser-id, or --launch to select the browser")
            if args.list_tabs:
                if not console.connection:
                    raise ValueError("No browser selected")
                print(json.dumps(console.connection.list_tabs(), indent=2))
                return 0
            if console.connection:
                console.use(args.tab)
            if task:
                report = console.get_engine().run(task)
                console.show_result(report)
                if any(step.get("interrupted") for step in report.get("results", [])):
                    return 130
                return 0 if report["status"] == "completed" else 3
            while True:
                try:
                    line = input("groundwork> ")
                    if not console.command(line):
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\nDetached. Browser and tabs remain open.")
                    break
                except Exception as exc:
                    print(f"Error: {exc}", file=sys.stderr)
            return 0
    except KeyboardInterrupt:
        print("\nCancelled; browser and tabs remain open.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
