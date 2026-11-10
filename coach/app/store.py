"""Local SQLite store: sessions, their metrics and findings, chat turns and plans.

The database is <data>/coach.db and each session's files live in <data>/sessions/<id>/.
Schema changes are appended to MIGRATIONS; PRAGMA user_version counts how many have run. A step is SQL text, or a
function that runs in the same transaction as its version bump, so a step that fails changes nothing.
Player labels are free text the user types ("Me", "Sam"). video_name is a display name only.
Besides the planned columns, sessions keeps `error`, the AnalysisError code of a failed analysis.
The metrics and findings rows copy the current revision's report.json. metrics holds numbers only: a reading the
analysis could not take is left out, and the report says so.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from coach.schema import report_from_dict

ID_PATTERN = re.compile(r"[0-9a-f]{16}")
ROLES = ("near", "far")
DEFAULT_LABELS = {"near": "Near player", "far": "Far player"}
MAX_LABEL_CHARS = 40
TO_WORK_ON = ("improve", "priority")


class RepairError(RuntimeError):
    """A saved report that migration 2 needs is missing or unreadable. The database was left as it was."""

    def __init__(self, data_dir: Path, session_id: str, revision: int, problem: str):
        super().__init__(
            f"The saved measures could not be rebuilt. Session {session_id}, revision {revision}: {problem}. "
            f"Nothing was changed. Put back {data_dir / 'sessions' / session_id / f'r{revision}' / 'report.json'}, "
            f'or delete that session with: python -m coach.app delete {session_id} --data-dir "{data_dir}"')
        self.session_id = session_id


def _project(db: sqlite3.Connection, session_id: str, report) -> None:
    """Replace one session's metrics and findings rows with those of `report`, leaving out missing readings."""
    db.execute("DELETE FROM metrics WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM findings WHERE session_id = ?", (session_id,))
    db.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)", [
        (session_id, role, m.name, float(m.value), m.unit, float(m.confidence))
        for role, group in report.metrics.items() for m in group if m.value is not None
    ])
    db.executemany("INSERT INTO findings VALUES (?, ?, ?, ?, ?)",
                   [(session_id, f.player.value, f.id, f.severity.value, f.headline) for f in report.findings])


def _rebuild_from_reports(db: sqlite3.Connection, sessions_dir: Path) -> None:
    """Migration 2. Older versions saved a practice-guide midpoint where nothing was seen, and the report loader now
    reads those as missing. Every analysed session's rows are copied again from its current report.json, so the
    midpoints leave the table and every observed reading stays, whatever its confidence."""
    for session_id, revision in db.execute("SELECT id, revision FROM sessions WHERE revision >= 0 ORDER BY rowid").fetchall():
        path = sessions_dir / session_id / f"r{revision}" / "report.json"
        try:
            report = report_from_dict(json.loads(path.read_text("utf-8")))
        except FileNotFoundError:
            raise RepairError(sessions_dir.parent, session_id, revision, "its report.json is missing") from None
        except Exception as exc:
            raise RepairError(sessions_dir.parent, session_id, revision, f"its report.json cannot be read ({exc})") from exc
        if report.video_id != session_id or report.revision != revision:
            raise RepairError(sessions_dir.parent, session_id, revision, "its report.json is for another session or revision")
        _project(db, session_id, report)


MIGRATIONS = (
    """
    CREATE TABLE sessions(
        id TEXT PRIMARY KEY,
        created REAL NOT NULL,
        video_name TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        tier TEXT,
        revision INTEGER NOT NULL DEFAULT -1,
        n_frames INTEGER,
        fps REAL,
        player_label_near TEXT NOT NULL DEFAULT 'Near player',
        player_label_far TEXT NOT NULL DEFAULT 'Far player',
        timings_json TEXT NOT NULL DEFAULT '{}',
        error TEXT
    );
    CREATE TABLE metrics(
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        player TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        unit TEXT NOT NULL,
        confidence REAL NOT NULL
    );
    CREATE TABLE findings(
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        player TEXT NOT NULL,
        finding_id TEXT NOT NULL,
        severity TEXT NOT NULL,
        headline TEXT NOT NULL
    );
    CREATE TABLE chat(
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        ts REAL NOT NULL,
        speaker TEXT NOT NULL,
        text TEXT NOT NULL
    );
    CREATE TABLE plans(
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        player TEXT NOT NULL,
        created REAL NOT NULL,
        json TEXT NOT NULL
    );
    CREATE INDEX metrics_by_session ON metrics(session_id);
    CREATE INDEX metrics_by_name ON metrics(name);
    CREATE INDEX findings_by_session ON findings(session_id);
    CREATE INDEX chat_by_session ON chat(session_id, ts);
    """,
    _rebuild_from_reports,
)


def new_id() -> str:
    return secrets.token_hex(8)


def check_id(value: Any) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("unknown session")
    return value


def clean_label(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("a player name must be text")
    label = " ".join(value.split())
    if not label:
        raise ValueError("a player name cannot be empty")
    if len(label) > MAX_LABEL_CHARS:
        raise ValueError(f"a player name can have at most {MAX_LABEL_CHARS} characters")
    return label


@dataclass(frozen=True)
class SessionRow:
    id: str
    created: float
    video_name: str
    source_hash: str
    tier: Optional[str]
    revision: int
    n_frames: Optional[int]
    fps: Optional[float]
    labels: dict[str, str]
    timings: dict[str, Any]
    error: Optional[str]
    to_work_on: int = 0

    @property
    def duration_s(self) -> Optional[float]:
        if self.n_frames and self.fps:
            return self.n_frames / self.fps
        return None


class Store:
    def __init__(self, data_dir: Path, *, repair: bool = True):
        """repair=False stops before migrations that rebuild rows from saved reports. Only deleting a session uses
        it, so a session whose report is lost can still be removed when that rebuild cannot finish."""
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / "coach.db", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._migrate(repair)
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _migrate(self, repair: bool) -> None:
        with self._lock:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise RuntimeError("this database was made by a newer version of the app")
            for number, step in enumerate(MIGRATIONS[version:], start=version + 1):
                if isinstance(step, str):
                    self._db.executescript(f"BEGIN; {step} PRAGMA user_version = {number}; COMMIT;")
                    continue
                if not repair:
                    return
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    step(self._db, self.sessions_dir)
                    self._db.execute(f"PRAGMA user_version = {number}")
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
                self._db.execute("COMMIT")

    @contextmanager
    def _write(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / check_id(session_id)

    def add_session(self, session_id: str, video_name: str, source_hash: str) -> None:
        with self._write() as db:
            db.execute(
                "INSERT INTO sessions(id, created, video_name, source_hash) VALUES (?, ?, ?, ?)",
                (check_id(session_id), time.time(), video_name, source_hash),
            )

    def get(self, session_id: str) -> Optional[SessionRow]:
        rows = self._read(_SESSION_SELECT + " WHERE s.id = ?", (check_id(session_id),))
        return _session(rows[0]) if rows else None

    def sessions(self) -> list[SessionRow]:
        return [_session(row) for row in self._read(_SESSION_SELECT + " ORDER BY s.created DESC, s.rowid DESC")]

    def commit_revision(self, session_id: str, report, timings: dict[str, Any]) -> None:
        """Make `report` the session's current revision and replace its metrics and findings. A reading this
        revision could not take removes the session's earlier value for it, rather than keeping an older one."""
        with self._write() as db:
            db.execute(
                "UPDATE sessions SET tier = ?, revision = ?, n_frames = ?, fps = ?, timings_json = ?, error = NULL"
                " WHERE id = ?",
                (report.tier.value, report.revision, report.n_frames, float(report.fps),
                 json.dumps(timings), check_id(session_id)),
            )
            _project(db, session_id, report)

    def set_timings(self, session_id: str, timings: dict[str, Any]) -> None:
        with self._write() as db:
            db.execute("UPDATE sessions SET timings_json = ? WHERE id = ?", (json.dumps(timings), check_id(session_id)))

    def fail(self, session_id: str, code: str, timings: dict[str, Any]) -> None:
        with self._write() as db:
            db.execute(
                "UPDATE sessions SET error = ?, timings_json = ? WHERE id = ?",
                (code, json.dumps(timings), check_id(session_id)),
            )

    def set_labels(self, session_id: str, near: Any, far: Any) -> dict[str, str]:
        labels = {"near": clean_label(near), "far": clean_label(far)}
        if labels["near"].casefold() == labels["far"].casefold():
            raise ValueError("give the two players different names")
        with self._write() as db:
            db.execute(
                "UPDATE sessions SET player_label_near = ?, player_label_far = ? WHERE id = ?",
                (labels["near"], labels["far"], check_id(session_id)),
            )
        return labels

    def add_turns(self, session_id: str, turns: Iterable[tuple[str, str]], *,
                  unless_coach_said: Iterable[str] = ()) -> bool:
        """Add the turns, unless the coach has already said one of `unless_coach_said` in this session. Returns
        whether they were added. The check and the write are one transaction, so a stop saved by another request
        after this one's earlier check is never missed. Turns are timed inside it, in the order they are saved."""
        session_id, turns, stops = check_id(session_id), list(turns), tuple(unless_coach_said)
        with self._write() as db:
            if _coach_said(db, session_id, stops):
                return False
            now = time.time()
            db.executemany("INSERT INTO chat VALUES (?, ?, ?, ?)",
                           [(session_id, now + i * 1e-6, speaker, text) for i, (speaker, text) in enumerate(turns)])
        return True

    def turns(self, session_id: str, limit: int = 12) -> list[tuple[str, str]]:
        """The last `limit` chat turns, oldest first, as (speaker, text)."""
        rows = self._read(
            "SELECT speaker, text FROM chat WHERE session_id = ? ORDER BY ts DESC, rowid DESC LIMIT ?",
            (check_id(session_id), limit),
        )
        return [(row["speaker"], row["text"]) for row in reversed(rows)]

    def coach_said(self, session_id: str, texts: Iterable[str]) -> bool:
        """True when any coach turn in the session's whole chat, not only its last turns, is exactly one of `texts`."""
        session_id, texts = check_id(session_id), tuple(texts)
        with self._lock:
            return _coach_said(self._db, session_id, texts)

    def add_plan(self, session_id: str, player: str, plan: dict[str, Any], *,
                 unless_coach_said: Iterable[str] = ()) -> Optional[str]:
        """Save the plan and return its id, unless the coach has already said one of `unless_coach_said` in this
        session: then nothing is saved and None is returned. The check and the save are one transaction."""
        session_id, stops, plan_id = check_id(session_id), tuple(unless_coach_said), new_id()
        with self._write() as db:
            if _coach_said(db, session_id, stops):
                return None
            db.execute("INSERT INTO plans VALUES (?, ?, ?, ?, ?)",
                       (plan_id, session_id, player, time.time(), json.dumps(plan)))
        return plan_id

    def plan(self, plan_id: str) -> Optional[dict[str, Any]]:
        rows = self._read("SELECT * FROM plans WHERE id = ?", (check_id(plan_id),))
        if not rows:
            return None
        row = rows[0]
        return {"id": row["id"], "session_id": row["session_id"], "player": row["player"],
                "created": row["created"], "plan": json.loads(row["json"])}

    def labels(self) -> list[str]:
        """Player names of analysed sessions, most recently used first, ties in alphabetical order."""
        rows = self._read(
            "SELECT player_label_near AS label, created FROM sessions WHERE revision >= 0"
            " UNION ALL SELECT player_label_far, created FROM sessions WHERE revision >= 0"
        )
        latest: dict[str, float] = {}
        for row in rows:
            latest[row["label"]] = max(latest.get(row["label"], 0.0), row["created"])
        return sorted(latest, key=lambda label: (-latest[label], label.casefold()))

    def progress(self, label: str, metric: Optional[str] = None) -> list[dict[str, Any]]:
        """Metric values of every analysed session where `label` names one of the players, oldest first."""
        sql = (
            "SELECT s.id, s.created, s.video_name, m.player, m.name, m.value, m.unit, m.confidence"
            " FROM sessions s JOIN metrics m ON m.session_id = s.id"
            " WHERE ((m.player = 'near' AND s.player_label_near = ?) OR (m.player = 'far' AND s.player_label_far = ?))"
        )
        args: tuple = (label, label)
        if metric is not None:
            sql += " AND m.name = ?"
            args += (metric,)
        rows = self._read(sql + " ORDER BY s.created, s.rowid, m.name", args)
        return [
            {"session_id": r["id"], "created": r["created"], "video_name": r["video_name"], "player": r["player"],
             "name": r["name"], "value": r["value"], "unit": r["unit"], "confidence": r["confidence"]}
            for r in rows
        ]

    def earlier(self, session_id: str, label: str, limit: int = 5) -> list[tuple[str, str]]:
        """Up to `limit` analysed sessions before this one where `label` played, oldest first, as (id, role)."""
        rows = self._read(
            "SELECT s.id, CASE WHEN s.player_label_near = ? THEN 'near' ELSE 'far' END AS role"
            " FROM sessions s, sessions me WHERE me.id = ? AND s.id != me.id AND s.revision >= 0"
            " AND (s.created < me.created OR (s.created = me.created AND s.rowid < me.rowid))"
            " AND (s.player_label_near = ? OR s.player_label_far = ?)"
            " ORDER BY s.created DESC, s.rowid DESC LIMIT ?",
            (label, check_id(session_id), label, label, limit),
        )
        return [(row["id"], row["role"]) for row in reversed(rows)]

    def delete(self, session_id: str) -> bool:
        """Delete one session's rows and files. False when there was nothing to delete."""
        folder = self.session_dir(session_id)
        with self._write() as db:
            gone = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount
        existed = folder.exists()
        if existed:
            shutil.rmtree(folder)
        return bool(gone) or existed


_SESSION_SELECT = (
    "SELECT s.*, (SELECT COUNT(*) FROM findings f WHERE f.session_id = s.id AND f.severity IN ({}))"
    " AS to_work_on FROM sessions s"
).format(", ".join(f"'{s}'" for s in TO_WORK_ON))


def _session(row: sqlite3.Row) -> SessionRow:
    return SessionRow(
        id=row["id"], created=row["created"], video_name=row["video_name"], source_hash=row["source_hash"],
        tier=row["tier"], revision=row["revision"], n_frames=row["n_frames"], fps=row["fps"],
        labels={"near": row["player_label_near"], "far": row["player_label_far"]},
        timings=json.loads(row["timings_json"]), error=row["error"], to_work_on=row["to_work_on"],
    )


def _coach_said(db: sqlite3.Connection, session_id: str, texts: tuple[str, ...]) -> bool:
    if not texts:
        return False
    return bool(db.execute(
        "SELECT EXISTS(SELECT 1 FROM chat WHERE session_id = ? AND speaker = 'coach' AND text IN"
        f" ({', '.join('?' * len(texts))}))",
        (session_id, *texts),
    ).fetchone()[0])
