"""Start the badminton coach on this computer, or delete one session.

    python -m coach.app [--port 8765] [--backend turbo|fast|replay] [--data-dir DIR] [--no-browser] [--no-auto-refine]
    python -m coach.app delete <session_id> [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import threading
import webbrowser
from pathlib import Path

from .backend import PROJECT, FastBackend, ReplayBackend, TurboBackend
from .server import App, Server, real_providers
from .store import RepairError, Store

DATA = PROJECT / "data"
BACKENDS = {"turbo": TurboBackend, "fast": FastBackend, "replay": lambda: ReplayBackend.for_samples(preview_stride=4)}
LABELS = {"turbo": TurboBackend.label, "fast": FastBackend.label, "replay": ReplayBackend.label}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["delete"]:
        return delete(argv[1:])
    return serve(argv)


def serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m coach.app", description="The badminton coach, on this computer only.")
    parser.add_argument("--port", type=int, default=8765, help="0 picks any free port (default 8765)")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="turbo",
                        help="turbo (default), fast (every frame, slower), or replay (stored pose files, for tests)")
    parser.add_argument("--data-dir", type=Path, default=DATA, help=f"where sessions are kept (default {DATA})")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--no-auto-refine", action="store_true", help="do not sharpen key moments by itself")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backend = args.backend
    if backend == "turbo" and not _installed("turbo.service"):
        print("Turbo is not installed here, so the slower Fast analysis is used.")
        backend = "fast"
    providers = real_providers()
    try:
        store = Store(args.data_dir)
    except RepairError as exc:
        print(exc, file=sys.stderr)
        return 3
    app = App(store, BACKENDS[backend], auto_refine=not args.no_auto_refine, **providers)
    try:
        server = Server(app, args.port)
    except OSError as exc:
        store.close()
        print(f"Port {args.port} is not free ({exc.strerror or exc}). Try --port 0 for any free port.", file=sys.stderr)
        return 2
    app.start()
    print(f"Badminton coach: {server.url}")
    print(f"Analysis: {LABELS[backend]}")
    print(f"Data: {Path(args.data_dir).resolve()}")
    missing = [name for name, provider in providers.items() if provider is None]
    if missing:
        print("Not installed yet: " + ", ".join(missing))
    sys.stdout.flush()
    threading.Thread(target=_report_warm_up, args=(app,), name="coach-warm-report", daemon=True).start()
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.server_close()
        app.stop()
        store.close()
    return 0


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _report_warm_up(app: App) -> None:
    warm = app.wait_until_warm()
    if warm["state"] == "ready":
        print(f"Analyser ready after {warm['seconds']:.1f} s of warm-up.", flush=True)
    elif warm["state"] == "failed":
        print(f"The analyser could not start after {warm['seconds']:.1f} s. See the log above.", flush=True)


def delete(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m coach.app delete", description="Delete one session and its files.")
    parser.add_argument("session_id")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args(argv)
    store = Store(args.data_dir, repair=False)
    try:
        gone = store.delete(args.session_id)
    except ValueError:
        print("That is not a session id.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"The session is deleted, but some of its files are still in use ({exc.strerror or exc}). "
              "Close the app and run this again to remove them.", file=sys.stderr)
        return 3
    finally:
        store.close()
    print("Deleted." if gone else "There is no session with that id.")
    return 0 if gone else 1


if __name__ == "__main__":
    sys.exit(main())
