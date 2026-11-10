import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from coach.app.store import MIGRATIONS, RepairError, Store, check_id, new_id
from coach.schema import report_from_dict
from coach.tests.fixture_report import earlier_report, full_report

APP_DATA = Path(__file__).with_name("app_data")
REPLAY = "20f02275bb919212"
REPLAY_SHA256 = {
    0: "0fd5401db770fa1e90ea9218bd1fe8513c4e67f1ed1a6921d2c1aa792a2810e3",
    1: "6c904767dcd3f93ab9692e4fc757e54109c426ea209dc573a4dc5c9dc451cfbf",
}
# Readings the saved Replay reports (schema 2) could not take: each holds a practice-guide midpoint with n_obs 0.
UNTAKEN = {
    0: {(role, name) for role in ("near", "far") for name in ("contact_reach", "contact_elbow", "swing_hand_speed")},
    1: {("far", "contact_reach"), ("far", "contact_elbow"), ("near", "ready_hip_ratio"), ("near", "ready_knee_flexion"),
        ("near", "ready_stance_width"), ("near", "ready_trunk_lean"), ("near", "contact_reach"), ("near", "contact_elbow")},
}
UNSURE, WAITING, FAILED = "0123456789abcdef", "1111111111111111", "2222222222222222"


def replay_bytes(revision: int) -> bytes:
    """A report.json the app saved for a Replay upload, checked against the hash of the file it wrote."""
    data = (APP_DATA / f"replay-{REPLAY}-r{revision}-report.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REPLAY_SHA256[revision]:
        raise AssertionError(f"coach/tests/app_data: the Replay r{revision} report changed")
    return data


def replay_report(revision: int):
    return report_from_dict(json.loads(replay_bytes(revision)))


def unsure_raw() -> dict:
    """A schema-2 report whose far hip-height reading was observed 12 times but with confidence 0."""
    raw = full_report().to_dict()
    raw.update(schema_version=2, video_id=UNSURE, revision=0)
    next(m for m in raw["metrics"]["far"] if m["name"] == "hip_height_ratio")["confidence"] = 0.0
    return raw


def legacy_rows(raw: dict) -> list[tuple]:
    """The metrics rows the app wrote before migration 2: every saved number, practice-guide midpoints included."""
    return [(role, m["name"], float(m["value"]), m["unit"], float(m["confidence"]))
            for role, group in raw["metrics"].items() for m in group]


def put_report(data: Path, session_id: str, revision: int, payload: bytes) -> None:
    path = data / "sessions" / session_id / f"r{revision}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def metric_rows(data: Path, session_id: str) -> list[tuple]:
    with closing(sqlite3.connect(data / "coach.db")) as db:
        return db.execute("SELECT player, name, value, unit, confidence FROM metrics WHERE session_id = ? ORDER BY rowid",
                          (session_id,)).fetchall()


def snapshot(data: Path) -> tuple[int, dict[str, list]]:
    """The schema version and every row of every table, read on a connection of its own."""
    with closing(sqlite3.connect(data / "coach.db")) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        return version, {table: db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                         for table in ("sessions", "metrics", "findings", "chat", "plans")}


def legacy_folder(data: Path) -> dict[str, list[tuple]]:
    """Fill `data` as the app left it before migration 2 and return each analysed session's metrics rows.

    REPLAY is at revision 1, whose report lost the near player's ready-stance readings; its r0 report is still on
    disk. UNSURE has an observed reading with confidence 0. WAITING and FAILED never got a revision."""
    unsure = unsure_raw()
    store = Store(data)
    try:
        store.add_session(REPLAY, "replay.mp4", "ab" * 32)
        for revision in (0, 1):
            put_report(data, REPLAY, revision, replay_bytes(revision))
            store.commit_revision(REPLAY, replay_report(revision), {"total": 0.5 + revision})
        store.set_labels(REPLAY, "Ana", "Ben")
        store.add_turns(REPLAY, [("user", "how do I get lower?"), ("coach", "Bend your knees more.")])
        store.add_plan(REPLAY, "near", {"weeks": [{"days": ["wall sit"]}]})
        store.add_session(UNSURE, "club.mp4", "cd" * 32)
        put_report(data, UNSURE, 0, json.dumps(unsure).encode("utf-8"))
        store.commit_revision(UNSURE, report_from_dict(unsure), {"total": 0.4})
        store.set_labels(UNSURE, "Ben", "Cy")
        store.add_session(WAITING, "new.mp4", "ef" * 32)
        store.add_session(FAILED, "dark.mp4", "01" * 32)
        store.fail(FAILED, "no_court", {"total": 0.2})
    finally:
        store.close()
    rows = {REPLAY: legacy_rows(json.loads(replay_bytes(1))), UNSURE: legacy_rows(unsure)}
    with closing(sqlite3.connect(data / "coach.db")) as db:
        for session_id, session_rows in rows.items():
            db.execute("DELETE FROM metrics WHERE session_id = ?", (session_id,))
            db.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)", [(session_id, *row) for row in session_rows])
        db.commit()
        db.execute("PRAGMA user_version = 1")
        db.commit()
    return rows


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.store = self.open()

    def open(self):
        store = Store(self.data)
        self.addCleanup(store.close)
        return store

    def sql(self, query, args=()):
        with closing(sqlite3.connect(self.data / "coach.db")) as db:
            return db.execute(query, args).fetchone()[0]

    def session(self, report=None, name="rally.mp4"):
        sid = new_id()
        self.store.add_session(sid, name, "ab" * 32)
        if report is not None:
            self.store.commit_revision(sid, report, {"total": 0.5})
        return sid

    def test_fresh_database_runs_every_migration_once(self):
        self.assertEqual(self.sql("PRAGMA user_version"), len(MIGRATIONS))
        self.assertEqual(self.sql("PRAGMA journal_mode"), "wal")
        sid = self.session()
        self.store.close()
        again = self.open()
        self.assertEqual(again.get(sid).video_name, "rally.mp4")

    def test_newer_database_is_refused(self):
        self.store.close()
        with closing(sqlite3.connect(self.data / "coach.db")) as db:
            db.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
            db.commit()
        with self.assertRaisesRegex(RuntimeError, "newer version"):
            Store(self.data)

    def test_new_session_waits_for_its_first_revision(self):
        sid = self.session()
        row = self.store.get(sid)
        self.assertEqual((row.revision, row.tier, row.error, row.to_work_on), (-1, None, None, 0))
        self.assertEqual(row.labels, {"near": "Near player", "far": "Far player"})

    def test_revision_replaces_metrics_and_findings(self):
        report = full_report()
        sid = self.session(report)
        self.store.commit_revision(sid, replace(report, revision=2), {"total": 1.0})
        row = self.store.get(sid)
        self.assertEqual((row.revision, row.tier, row.n_frames, row.fps), (2, "refined", 900, 30.0))
        self.assertEqual(row.duration_s, 30.0)
        self.assertEqual(row.timings, {"total": 1.0})
        n_metrics = self.sql("SELECT COUNT(*) FROM metrics WHERE session_id = ?", (sid,))
        n_findings = self.sql("SELECT COUNT(*) FROM findings WHERE session_id = ?", (sid,))
        self.assertEqual(n_metrics, sum(len(group) for group in report.metrics.values()))
        self.assertEqual(n_findings, 32)
        expected = sum(1 for f in report.findings if f.severity.value in ("improve", "priority"))
        self.assertEqual(row.to_work_on, expected)
        self.assertGreater(expected, 0)

    def test_labels_are_cleaned_and_must_differ(self):
        sid = self.session()
        self.assertEqual(self.store.set_labels(sid, "  Me \t", "Sam"), {"near": "Me", "far": "Sam"})
        with self.assertRaisesRegex(ValueError, "different names"):
            self.store.set_labels(sid, "Sam", "sam")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.store.set_labels(sid, " ", "Sam")
        with self.assertRaisesRegex(ValueError, "at most"):
            self.store.set_labels(sid, "x" * 41, "Sam")
        self.assertEqual(self.store.get(sid).labels, {"near": "Me", "far": "Sam"})

    def test_hostile_text_is_stored_as_plain_text(self):
        sid = self.session(name="x'); DROP TABLE sessions;--<b>.mp4")
        self.store.add_turns(sid, [("user", "'); DELETE FROM chat; --"), ("coach", "<img src=x onerror=alert(1)>")])
        self.assertEqual(self.store.get(sid).video_name, "x'); DROP TABLE sessions;--<b>.mp4")
        self.assertEqual(self.store.turns(sid), [("user", "'); DELETE FROM chat; --"),
                                                  ("coach", "<img src=x onerror=alert(1)>")])

    def test_turns_returns_the_last_twelve_oldest_first(self):
        sid = self.session()
        self.store.add_turns(sid, [("user" if i % 2 == 0 else "coach", f"turn {i}") for i in range(15)])
        turns = self.store.turns(sid)
        self.assertEqual(len(turns), 12)
        self.assertEqual(turns[0], ("coach", "turn 3"))
        self.assertEqual(turns[-1], ("user", "turn 14"))

    def test_coach_said_looks_at_the_whole_chat_for_the_coach_s_exact_words(self):
        sid, other = self.session(), self.session()
        self.store.add_turns(sid, [("user", "it hurts"), ("coach", "Stop here.")])
        self.store.add_turns(sid, [("user" if i % 2 == 0 else "coach", f"turn {i}") for i in range(20)])
        self.store.add_turns(other, [("user", "Stop here."), ("coach", "Stop here, please.")])
        self.assertTrue(self.store.coach_said(sid, ["Stop", "Stop here."]))
        self.assertFalse(self.store.coach_said(sid, ["it hurts", "stop here."]))
        self.assertFalse(self.store.coach_said(sid, []))
        self.assertFalse(self.store.coach_said(other, ["Stop here."]))

    def test_a_write_that_is_refused_for_a_stop_saves_nothing(self):
        sid, other = self.session(), self.session()
        stops = ["Stop", "Stop here."]
        self.assertIs(self.store.add_turns(sid, [("user", "a drill?"), ("coach", "Try lunges.")],
                                           unless_coach_said=stops), True)
        kept = self.store.add_plan(sid, "near", {"weeks": [1]}, unless_coach_said=stops)
        self.assertIsNotNone(kept)
        self.store.add_turns(sid, [("user", "it hurts"), ("coach", "Stop here.")])
        before = snapshot(self.data)
        self.assertIs(self.store.add_turns(sid, [("user", "a drill?"), ("coach", "Try lunges.")],
                                           unless_coach_said=stops), False)
        self.assertIsNone(self.store.add_plan(sid, "far", {"weeks": [2]}, unless_coach_said=stops))
        self.assertEqual(snapshot(self.data), before)
        # A write without the condition is still saved, and a player saying those words stops nothing.
        self.assertIs(self.store.add_turns(sid, [("user", "and now?"), ("coach", "Stop here.")]), True)
        self.assertIs(self.store.add_turns(other, [("user", "Stop here."), ("coach", "Try lunges.")],
                                           unless_coach_said=stops), True)
        self.assertIsNotNone(self.store.add_plan(other, "near", {"weeks": [3]}, unless_coach_said=stops))
        self.assertEqual(self.store.turns(sid, 100), [("user", "a drill?"), ("coach", "Try lunges."),
                                                      ("user", "it hurts"), ("coach", "Stop here."),
                                                      ("user", "and now?"), ("coach", "Stop here.")])
        self.assertEqual(self.store.plan(kept)["plan"], {"weeks": [1]})

    def test_progress_follows_the_label_across_roles(self):
        first = self.session(earlier_report())
        self.store.set_labels(first, "Me", "Sam")
        second = self.session(full_report())
        self.store.set_labels(second, "Sam", "Me")
        rows = self.store.progress("Me", "hip_height_ratio")
        self.assertEqual([(r["session_id"], r["player"], r["value"]) for r in rows],
                         [(first, "near", 0.56), (second, "far", 0.45)])
        self.assertEqual(self.store.progress("Nobody", "hip_height_ratio"), [])
        names = {r["name"] for r in self.store.progress("Me")}
        self.assertIn("hip_height_ratio", names)
        third = self.session(full_report())
        self.store.set_labels(third, "Kim", "Sam")
        self.assertEqual(self.store.labels(), ["Kim", "Sam", "Me"])

    def test_earlier_sessions_are_oldest_first_and_capped(self):
        ids = []
        for _ in range(7):
            sid = self.session(full_report())
            self.store.set_labels(sid, "Me", "Sam")
            ids.append(sid)
        unanalysed = self.session()
        self.store.set_labels(unanalysed, "Me", "Sam")
        later = ids[-1]
        self.assertEqual(self.store.earlier(later, "Me"), [(sid, "near") for sid in ids[1:6]])
        self.assertEqual(self.store.earlier(ids[0], "Me"), [])
        self.assertEqual(self.store.earlier(later, "Sam", limit=2), [(sid, "far") for sid in ids[4:6]])

    def test_plans_round_trip(self):
        sid = self.session()
        plan_id = self.store.add_plan(sid, "near", {"weeks": [{"days": []}]})
        saved = self.store.plan(plan_id)
        self.assertEqual((saved["session_id"], saved["player"], saved["plan"]), (sid, "near", {"weeks": [{"days": []}]}))

    def test_delete_removes_rows_and_files(self):
        sid = self.session(full_report())
        self.store.add_turns(sid, [("user", "hi")])
        self.store.add_plan(sid, "near", {})
        folder = self.store.session_dir(sid)
        folder.mkdir()
        (folder / "video.mp4").write_bytes(b"x")
        self.assertTrue(self.store.delete(sid))
        self.assertIsNone(self.store.get(sid))
        self.assertFalse(folder.exists())
        for table in ("metrics", "findings", "chat", "plans"):
            count = self.sql(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (sid,))
            self.assertEqual(count, 0, table)
        self.assertFalse(self.store.delete(sid))

    def test_ids_that_could_leave_the_folder_are_refused(self):
        for bad in ("..\\..\\coach", "../../x", "%2e%2e%2f", "0123456789abcdeF", "0123456789abcde", "", None, 7):
            with self.assertRaises(ValueError):
                check_id(bad)
            with self.assertRaises(ValueError):
                self.store.session_dir(bad)

    def test_a_reading_the_new_revision_could_not_take_leaves_no_row(self):
        self.store.add_session(REPLAY, "replay.mp4", "ab" * 32)
        self.store.set_labels(REPLAY, "Ana", "Ben")
        self.store.commit_revision(REPLAY, replay_report(0), {"total": 0.5})
        rows = metric_rows(self.data, REPLAY)
        self.assertEqual(len(rows), 18)
        self.assertEqual({row[:2] for row in rows} & UNTAKEN[0], set())
        self.assertIn(("near", "ready_hip_ratio", 0.5509120690586571, "heights", 0.55), rows)
        self.assertEqual([(r["player"], r["value"]) for r in self.store.progress("Ana", "ready_hip_ratio")],
                         [("near", 0.5509120690586571)])

        self.store.commit_revision(REPLAY, replay_report(1), {"total": 0.9})
        rows = metric_rows(self.data, REPLAY)
        self.assertEqual(len(rows), 28)
        self.assertEqual({row[:2] for row in rows} & UNTAKEN[1], set())
        self.assertEqual(self.store.progress("Ana", "ready_hip_ratio"), [])
        self.assertEqual([(r["player"], r["value"]) for r in self.store.progress("Ben", "ready_hip_ratio")],
                         [("far", 0.5195009686061087)])

    def test_an_observed_reading_with_confidence_zero_is_kept(self):
        report = full_report()
        far = tuple(replace(m, confidence=0.0) if m.name == "hip_height_ratio" else m for m in report.metrics["far"])
        sid = self.session(replace(report, metrics={**report.metrics, "far": far}))
        self.assertIn(("far", "hip_height_ratio", 0.45, "heights", 0.0), metric_rows(self.data, sid))
        self.assertEqual(len(metric_rows(self.data, sid)), 33)


class MigrationTest(unittest.TestCase):
    """Migration 2 on a data folder the app wrote before it: user_version 1 and the old metrics rows."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        self.legacy = legacy_folder(self.data)
        self.report_path = self.data / "sessions" / REPLAY / "r1" / "report.json"

    def open(self, **options):
        store = Store(self.data, **options)
        self.addCleanup(store.close)
        return store

    def assert_rebuilt(self):
        self.assertEqual(snapshot(self.data)[0], 2)
        replay = metric_rows(self.data, REPLAY)
        self.assertEqual(len(replay), 28)
        self.assertEqual(replay, [row for row in self.legacy[REPLAY] if row[:2] not in UNTAKEN[1]])
        self.assertIn(("far", "ready_hip_ratio", 0.5195009686061087, "heights", 0.55), replay)
        unsure = metric_rows(self.data, UNSURE)
        self.assertEqual(unsure, self.legacy[UNSURE])
        self.assertIn(("far", "hip_height_ratio", 0.45, "heights", 0.0), unsure)

    def test_rebuild_drops_untaken_readings_and_keeps_everything_else(self):
        version, before = snapshot(self.data)
        self.assertEqual(version, 1)
        self.assertEqual(len(metric_rows(self.data, REPLAY)), 36)
        self.assertIn(("near", "ready_hip_ratio", 0.45, "heights", 0.0), metric_rows(self.data, REPLAY))
        store = self.open()
        self.assert_rebuilt()
        after = snapshot(self.data)[1]
        for table in ("sessions", "findings", "chat", "plans"):
            self.assertEqual(after[table], before[table], table)
        self.assertEqual(store.get(REPLAY).labels, {"near": "Ana", "far": "Ben"})
        self.assertEqual(store.turns(REPLAY), [("user", "how do I get lower?"), ("coach", "Bend your knees more.")])
        self.assertEqual(store.progress("Ana", "ready_hip_ratio"), [])
        self.assertEqual([(r["session_id"], r["player"], r["value"]) for r in store.progress("Ben", "ready_hip_ratio")],
                         [(REPLAY, "far", 0.5195009686061087)])
        self.assertEqual(store.get(WAITING).revision, -1)
        self.assertEqual(store.get(FAILED).error, "no_court")
        for revision in (0, 1):
            path = self.data / "sessions" / REPLAY / f"r{revision}" / "report.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), REPLAY_SHA256[revision])

    def test_running_the_rebuild_again_gives_the_same_rows(self):
        self.open().close()
        rebuilt = snapshot(self.data)
        self.open().close()
        self.assertEqual(snapshot(self.data), rebuilt)
        with closing(sqlite3.connect(self.data / "coach.db")) as db:
            db.execute("PRAGMA user_version = 1")
            db.commit()
        self.open().close()
        self.assertEqual(snapshot(self.data), rebuilt)

    def test_a_rebuild_that_cannot_finish_changes_nothing_and_runs_again_once_fixed(self):
        # UNSURE is rebuilt after REPLAY, so each failure comes after REPLAY's rows were already rewritten.
        path = self.data / "sessions" / UNSURE / "r0" / "report.json"
        good = path.read_bytes()
        contradictory = unsure_raw()
        next(m for m in contradictory["metrics"]["near"] if m["name"] == "hip_height_ratio").update(n_obs=0, confidence=0.3)
        problems = {
            "its report.json is missing": None,
            "its report.json cannot be read (Expecting property name enclosed in double quotes: line 1 column 22"
            " (char 21))": b'{"schema_version": 2,',
            "its report.json is for another session or revision": replay_bytes(1),
            "its report.json cannot be read (metric hip_height_ratio has n_obs 0 but confidence or best is set;"
            " refusing to guess)": json.dumps(contradictory).encode("utf-8"),
        }
        legacy = snapshot(self.data)
        for problem, payload in problems.items():
            with self.subTest(problem=problem):
                if payload is None:
                    path.unlink()
                else:
                    path.write_bytes(payload)
                with self.assertRaises(RepairError) as caught:
                    Store(self.data)
                self.assertEqual(str(caught.exception), (
                    f"The saved measures could not be rebuilt. Session {UNSURE}, revision 0: {problem}. Nothing was "
                    f"changed. Put back {path}, or delete that session with: "
                    f'python -m coach.app delete {UNSURE} --data-dir "{self.data}"'))
                self.assertEqual(caught.exception.session_id, UNSURE)
                self.assertEqual(snapshot(self.data), legacy)
                self.assertEqual(len(metric_rows(self.data, REPLAY)), 36)
        path.write_bytes(good)
        self.open()
        self.assert_rebuilt()

    def test_a_session_whose_report_is_lost_can_still_be_deleted(self):
        self.report_path.unlink()
        with self.assertRaises(RepairError):
            Store(self.data)
        store = Store(self.data, repair=False)
        try:
            self.assertEqual(snapshot(self.data)[0], 1)
            self.assertTrue(store.delete(REPLAY))
        finally:
            store.close()
        self.assertFalse((self.data / "sessions" / REPLAY).exists())
        store = self.open()
        self.assertEqual(snapshot(self.data)[0], 2)
        self.assertIsNone(store.get(REPLAY))
        self.assertEqual(metric_rows(self.data, UNSURE), self.legacy[UNSURE])
        self.assertEqual(store.get(UNSURE).labels, {"near": "Ben", "far": "Cy"})


if __name__ == "__main__":
    unittest.main()
