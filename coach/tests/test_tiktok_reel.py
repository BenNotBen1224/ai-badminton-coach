from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from coach.reel_media import RATE, checked_voice_log, compose, schedule, verify
from coach.reel_analysis import find_hits, rally_windows
from coach.reel_plan import FOOTER, draft_plan, make_plan, sequence_context, validate_plan, wording
from coach.tiktok_reel import credit
from scripts.evaluate_reel_analysis import score


def plan():
    return {
        "schema_version": 1, "review_status": "automatic_unreviewed",
        "source_sha256": "source", "analysis_fps": 30,
        "names_at_first_reliable_rally": {"near": "甲", "far": "乙"},
        "title": ["甲对上乙", "这几拍，攻防节奏拉满！"], "footer": FOOTER,
        "attribution": {"author": "A", "license": "CC BY 4.0", "source_url": "https://example.com",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/"},
        "target_frames": 798,
        "moments": [{
            "start_frame": 30, "end_frame": 210,
            "lines": [
                {"cue_frame": 45, "hitter": "near", "player": "甲", "say": "甲对角调动！",
                 "caption": "甲对角调动！", "caption_chunks": ["甲对角调动！"], "tag": "飞行约 0.8 秒 · 估算"},
                {"cue_frame": 120, "hitter": "far", "player": "乙", "say": "乙压底角！",
                 "caption": "乙压底角！", "caption_chunks": ["乙压底角！"], "tag": "飞行约 1.2 秒 · 估算"},
            ],
        }],
    }


class PlanTests(unittest.TestCase):
    def test_deep_then_short_context_is_derived_from_placements_not_line_index(self):
        first = {"cue_frame": 30, "hitter": "far", "player": "甲", "shot_family": None,
                 "evidence": {"receiver_court_m": [3., 12.], "flight_s": 1.2}}
        second = {"cue_frame": 90, "hitter": "far", "player": "甲", "shot_family": None,
                  "evidence": {"receiver_court_m": [3., 7.5], "flight_s": .8}}
        sequence_context([{"lines": [first, second]}])
        self.assertEqual(first["say"], "甲先压后场！")
        self.assertEqual(second["say"], "甲再收网前，前后调动！")
        self.assertEqual(second["sequence_evidence"]["earlier_cue_frame"], 30)
        self.assertIsNone(second["shot_family"])

    def test_analysis_draft_is_editable_but_not_silently_renderable(self):
        doc = draft_plan({"frames": 900, "hits": []}, {"near": "甲", "far": "乙"},
                         plan()["attribution"], "source", {})
        self.assertEqual(doc["moments"], [])
        self.assertEqual(doc["moment_template"]["lines"][0]["claim_source"], "reviewed_plan")
        with self.assertRaisesRegex(ValueError, "Complete the draft"):
            validate_plan(doc, "source", 900)
        doc["moments"] = plan()["moments"]
        doc["review_status"] = "reviewed_plan"
        doc["reviewer_kind"] = "agent"
        doc["reviewer"] = "Test fixture"
        doc["review_note"] = "Reviewed source frames and wording."
        for line in doc["moments"][0]["lines"]:
            line["claim_source"] = "reviewed_plan"
            line["evidence"] = {"review_note": "Visible hitter, court position and flight reviewed."}
        validate_plan(doc, "source", 900)
        doc["moments"][0]["lines"][0]["evidence"]["review_note"] = ""
        with self.assertRaisesRegex(ValueError, "visual evidence"):
            validate_plan(doc, "source", 900)

    def test_missing_or_empty_hook_title_is_rejected(self):
        for title in ([], ["甲对上乙"], ["甲对上乙", ""]):
            doc = plan()
            doc["title"] = title
            with self.assertRaisesRegex(ValueError, "two-line title"):
                validate_plan(doc, "source", 900)

    def test_wrong_source_and_out_of_bounds_window_fail(self):
        with self.assertRaisesRegex(ValueError, "different source"):
            validate_plan(plan(), "other", 900)
        doc = plan()
        doc["moments"][0]["end_frame"] = 900
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_plan(doc, "source", 900)

    def test_missing_caption_and_wrong_caption_chunks_fail(self):
        doc = plan()
        doc["moments"][0]["lines"][0]["caption_chunks"] = ["different"]
        with self.assertRaisesRegex(ValueError, "chunks"):
            validate_plan(doc, "source", 900)
        doc["moments"][0]["lines"][0]["caption"] = ""
        with self.assertRaisesRegex(ValueError, "caption"):
            validate_plan(doc, "source", 900)

    def test_spoken_numbers_and_unmarked_estimates_fail(self):
        doc = plan()
        doc["moments"][0]["lines"][0]["say"] = "甲跑了三米！"
        with self.assertRaisesRegex(ValueError, "spoken numbers"):
            validate_plan(doc, "source", 900)
        doc = plan()
        doc["moments"][0]["lines"][0]["tag"] = "跑了 3 米"
        with self.assertRaisesRegex(ValueError, "estimates"):
            validate_plan(doc, "source", 900)

    def test_normal_chinese_count_words_and_approved_phrasing_remain_editable(self):
        for text in ("这一拍，甲对角调动！", "两人头顶对角互拉！", "甲打四方球，前后调动！"):
            doc = plan()
            doc["moments"][0]["lines"][0].update(say=text, caption=text, caption_chunks=[text])
            validate_plan(doc, "source", 900)
        for text in ("甲半秒就到！", "甲跑近四米！", "甲飞行1.2秒！", "甲成功率百分之八十！"):
            doc = plan()
            doc["moments"][0]["lines"][0]["say"] = text
            with self.assertRaisesRegex(ValueError, "measured stats"):
                validate_plan(doc, "source", 900)

    def test_no_rally_fails_instead_of_fabricating_highlights(self):
        with self.assertRaisesRegex(ValueError, "No reliable"):
            make_plan({"rallies": [], "hits": []}, {}, Path("unused"),
                      {"near": "甲", "far": "乙"}, {}, "source")

    def test_unknown_family_uses_no_smash_or_backhand_claim(self):
        hit = {"family": None, "hitter": "near", "hitter_court_m": [3., 12.]}
        self.assertIsNone(wording(hit, "甲"))
        hit["receiver_court_m"] = [.2, 1.]
        self.assertEqual(wording(hit, "甲"), ("甲对角调动！", "对角调动"))
        hit["family"] = "lift"
        self.assertEqual(wording(hit, "甲"), ("甲挑后场，拆连贯！", "挑后场"))

    def test_cleaning_contacts_outside_rallies_do_not_anchor_identity(self):
        hit = {"frame": 45, "time_s": 1.5, "family": None, "next_hit_frame": 75,
               "visible_fraction": .9, "confidence": .8, "hitter": "near",
               "hitter_court_m": [3., 12.], "receiver_court_m": [.2, 1.], "flight_s": 1.}
        other = {**hit, "frame": 105, "time_s": 3.5, "hitter": "far"}
        analysis = {"rallies": [{"start_s": 1., "end_s": 5., "hit_frames": [45, 105]}],
                    "hits": [{**hit, "frame": 5}, hit, other], "cuts": [], "frames": 900}
        assignments = {i: {"near": "甲", "far": "乙"} for i in (45, 105)}
        with patch("coach.reel_plan.identities", return_value=assignments) as identify:
            result = make_plan(analysis, {}, Path("unused"), {"near": "甲", "far": "乙"}, {"author": "A"}, "source")
        self.assertEqual([h["frame"] for h in identify.call_args.args[2]], [45, 105])
        self.assertEqual(result["moments"][0]["lines"][0]["player"], "甲")


class SpeechScheduleTests(unittest.TestCase):
    def test_unknown_phonemes_fail_including_windows_native_logs(self):
        checked = checked_voice_log(b"load warning\r\n@@VOICE 0\r\n@@ENDVOICE 0\r\n", 1)
        self.assertEqual(checked["dropped_phonemes"], 0)
        self.assertFalse(checked["human_listening"])
        for warning in (b"Skip unknown phonemes", b"Unknown token"):
            with self.assertRaisesRegex(ValueError, "dropped unknown"):
                checked_voice_log(b"@@VOICE 0\n"+warning+b"\n@@ENDVOICE 0\n", 1)
        with self.assertRaisesRegex(ValueError, "did not report"):
            checked_voice_log(b"@@VOICE 0\n", 1)

    def test_variable_voice_durations_fit_by_construction(self):
        for duration in (1.8346, 1.86, 2.0583, 3.5):
            voices = [np.full(round(duration*RATE), .1, np.float32)]*2
            rows, captions, audio = schedule(plan(), voices)
            self.assertEqual(len(rows), 798)
            self.assertEqual(len(audio), 638400)
            for caption in captions:
                self.assertLessEqual(caption["voice_start_frame"]/30+caption["voice_duration_s"],
                                     caption["end_frame"]/30)
                self.assertEqual(caption["voice_start_frame"]-caption["cue_output_frame"], 2)
                self.assertEqual(rows[caption["cue_output_frame"]]["source_frame"], caption["cue_frame"])
            self.assertEqual(rows[0]["source_frame"], 30)
            self.assertEqual(rows[-1]["source_frame"], 210)

    def test_invalid_long_edited_voice_fails_before_encoding(self):
        voices = [np.full(20*RATE, .1, np.float32)]*2
        with self.assertRaisesRegex(ValueError, "exceeds 30"):
            schedule(plan(), voices)

    def test_each_voice_slot_has_the_requested_actual_samples(self):
        voices = [np.full(RATE, .12, np.float32), np.full(2*RATE, -.23, np.float32)]
        _, captions, mix = schedule(plan(), voices)
        for row, expected in zip(captions, voices):
            start = round(row["voice_start_frame"]/30*RATE)
            np.testing.assert_array_equal(mix[start:start+len(expected)], expected)


class AttributionTests(unittest.TestCase):
    def test_local_rights_are_explicit_and_do_not_require_invented_urls(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"credit.json"
            for kind in ("self_recorded", "private_permission"):
                doc = {"author": "Local recorder", "license": "Private use by permission",
                       "source_kind": kind, "rights_statement": "I have permission to process this recording."}
                path.write_text(json.dumps(doc))
                result = credit(path)
                self.assertEqual(result["source_kind"], kind)
                self.assertEqual(result["rights_statement"], doc["rights_statement"])
                self.assertNotIn("source_url", result)
                self.assertNotIn("license_url", result)
                del doc["rights_statement"]
                path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, "rights_statement"):
                    credit(path)

    def test_external_urls_and_optional_local_urls_must_be_valid_https(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"credit.json"
            doc = plan()["attribution"].copy()
            path.write_text(json.dumps(doc))
            self.assertEqual(credit(path)["source_kind"], "external")
            for kind in ("external", "self_recorded", "private_permission"):
                for bad in ("http://example.com", "https://", "https:///missing-host", "https://example.com/a b",
                            "https://user:secret@example.com"):
                    path.write_text(json.dumps({**doc, "source_kind": kind, "source_url": bad,
                                                "rights_statement": "Permission supplied by fixture."}))
                    with self.assertRaisesRegex(ValueError, "HTTPS"):
                        credit(path)

    def test_unknown_source_cannot_inherit_sample_rights(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"credit.json"
            path.write_text(json.dumps({"author": "A"}))
            with self.assertRaisesRegex(ValueError, "Attribution needs"):
                credit(path)


class ScopedAgreementTests(unittest.TestCase):
    def test_wrong_prediction_changes_agreement_without_inventing_false_positives(self):
        labels = {"labeler": "AI fixture", "rallies": [{"id": "test", "hits": [{
            "t": 2., "hitter": "far", "family": "clear", "shuttle_seen": True,
            "conf": {"family": "high", "hitter": "high", "t": "med"},
        }]}]}
        predictions = {"hits": [{"time_s": 2.1, "hitter": "far", "family": "clear"},
                                {"time_s": 10., "hitter": "near", "family": "lift"}]}
        self.assertEqual(score(predictions, labels)["matched_eligible_family_agreement"], 1.)
        predictions["hits"][0]["family"] = "lift"
        result = score(predictions, labels)
        self.assertEqual(result["matched_eligible_family_agreement"], 0.)
        self.assertNotIn("precision", result)
        self.assertIn("Not scored as false positives", result["unmatched_predictions"])
        predictions["hits"][0]["family"] = None
        result = score(predictions, labels)
        self.assertEqual(result["matched_eligible_abstentions"], 1)
        self.assertIsNone(result["matched_eligible_family_agreement"])
        labels["rallies"][0]["hits"][0]["conf"]["family"] = "low"
        self.assertEqual(score(predictions, labels)["eligible_annotations"], 0)


class ContactEvidenceTests(unittest.TestCase):
    def test_perspective_overlapping_torsos_use_measured_racket_side(self):
        def player(box, shoulder_y, wrist, court):
            joints = [[100., shoulder_y, 1.] for _ in range(17)]
            joints[9] = joints[10] = [*wrist, 1.]
            return {"bbox_xyxy": box, "joints": joints, "court_m": court}
        near = player([50, 180, 200, 400], 210, [160, 280], [3, 8])
        far = player([70, 170, 140, 270], 200, [100, 190], [3, 5])
        frames = [{"near": near, "far": far} for _ in range(21)]
        points = [[100-abs(i-10)*10, 190, .99] for i in range(21)]
        hits = find_hits(points, frames, [])
        self.assertEqual([(h["frame"], h["hitter"], h["family"]) for h in hits], [(10, "far", None)])

    def test_player_motion_without_shuttle_never_makes_contacts(self):
        self.assertEqual(find_hits([None]*90, [{"near": {}, "far": {}}]*90, []), [])

    def test_a_shuttle_without_two_players_never_makes_contacts(self):
        points = [[i*4, i*3, .99] for i in range(90)]
        self.assertEqual(find_hits(points, [{}]*90, []), [])

    def test_same_side_repeated_motion_is_not_a_rally(self):
        hits = [{"frame": i, "time_s": i/30, "hitter": "near"} for i in (0, 30, 60, 90)]
        self.assertEqual(rally_windows(hits), [])

    def test_alternating_contacts_split_at_a_long_gap(self):
        hits = [{"frame": i*30, "time_s": i, "hitter": "near" if i%2 else "far"}
                for i in (0, 1, 2, 3, 10, 11)]
        self.assertEqual(rally_windows(hits), [{"start_s": 0, "end_s": 3.7,
                                              "hit_frames": [0, 30, 60, 90]}])

    def test_actual_camera_cut_splits_otherwise_linked_play(self):
        hits = [{"frame": i*30, "time_s": i, "hitter": "near" if i%2 else "far"}
                for i in range(6)]
        self.assertEqual(rally_windows(hits, [75]), [])


class OutputSafetyTests(unittest.TestCase):
    def test_existing_output_is_byte_for_byte_unchanged(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"old-output"
            out.mkdir()
            (out/"failure.json").write_bytes(b"previous receipt\n")
            (out/"plan.json").write_bytes(b"previous plan\n")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
            process = subprocess.run(
                [sys.executable, "-B", "-m", "coach.tiktok_reel", str(Path(temp)/"missing.mp4"), str(out)],
                capture_output=True)
            self.assertEqual(process.returncode, 1)
            self.assertIn(b"previous run left unchanged", process.stderr)
            self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}, before)

    def test_non_widescreen_picture_is_letterboxed_not_stretched(self):
        font = Path(os.environ.get("WINDIR", r"C:\Windows"))/"Fonts/msyh.ttc"
        if not font.exists():
            self.skipTest("Chinese font not installed")
        raw = np.full((480, 640, 3), (10, 30, 240), np.uint8)
        doc = plan()
        line = doc["moments"][0]["lines"][0]
        result = compose(raw, {"players": []}, doc, line, font)
        self.assertEqual(result.shape, (1920, 1080, 3))
        picture = result[350:951, 6:1074]
        coloured = np.all(picture == (10, 30, 240), axis=2)
        ys, xs = np.nonzero(coloured)
        self.assertEqual(int(ys.max()-ys.min()+1), 601)
        self.assertEqual(int(xs.max()-xs.min()+1), 801)
        self.assertFalse(coloured[:, 0].any())
        self.assertFalse(coloured[:, -1].any())


class EncodedVerificationTests(unittest.TestCase):
    def test_encoded_missing_title_tag_caption_and_muted_voice_are_rejected(self):
        import cv2
        import wave
        from coach.export import _tools
        font = Path(os.environ.get("WINDIR", r"C:\Windows"))/"Fonts/msyh.ttc"
        if not font.exists():
            self.skipTest("Chinese font not installed")
        ffmpeg, _ = _tools()
        doc, pose = plan(), {"players": []}
        line = doc["moments"][0]["lines"][0]
        raw = np.full((480, 640, 3), 100, np.uint8)
        picture = compose(raw, pose, doc, line, font)
        no_caption = compose(raw, pose, doc, line, font, False)
        count = 15
        rows = [{"source_frame": 45, "speech": 0} for _ in range(count)]
        captions = [{"voice_start_frame": 2, "voice_duration_s": .3, "cue_output_frame": 0, "cue_frame": 45}]
        audio = np.zeros(RATE//2, np.float32)
        start, size = RATE*2//30, round(RATE*.3)
        audio[start:start+size] = .2*np.sin(2*np.pi*440*np.arange(size)/RATE)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            references = folder/"references.npz"
            np.savez(references,
                     picture=np.repeat(cv2.resize(picture, (108, 192), interpolation=cv2.INTER_AREA)[None], count, axis=0),
                     caption=np.repeat(cv2.resize(picture[1175:1305], (270, 65), interpolation=cv2.INTER_AREA)[None], count, axis=0),
                     no_caption=np.repeat(cv2.resize(no_caption[1175:1305], (270, 65), interpolation=cv2.INTER_AREA)[None], count, axis=0),
                     audio=audio)
            for kind, expected_error in (("valid", None), ("title", "Encoded title"),
                                         ("green tag", "Encoded green tag"), ("caption", "white spoken captions"),
                                         ("voice", "silent")):
                encoded_picture = picture.copy()
                if kind in ("title", "green tag", "caption"):
                    top, bottom = {"title": (140, 285), "green tag": (1060, 1150), "caption": (1175, 1305)}[kind]
                    encoded_picture[top:bottom] = (29, 22, 7)
                wave_path = folder/f"{kind}.wav"
                with wave.open(str(wave_path), "wb") as stream:
                    stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                    stream.writeframes(np.rint((audio if kind != "voice" else audio*0)*32767).astype("<i2").tobytes())
                movie = folder/f"{kind}.mp4"
                subprocess.run([str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", "1080x1920", "-r", "30", "-i", "pipe:0", "-i", str(wave_path),
                                "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "-c:a", "aac", str(movie)],
                               input=encoded_picture.tobytes()*count, capture_output=True, check=True)
                if expected_error:
                    with self.assertRaisesRegex(ValueError, expected_error):
                        verify(movie, references, rows, captions, ffmpeg)
                else:
                    with np.load(references) as archive:
                        archive_type = type(archive)
                    getitem = archive_type.__getitem__
                    reads = {}
                    def counted_read(archive, key):
                        reads[key] = reads.get(key, 0)+1
                        return getitem(archive, key)
                    with patch.object(archive_type, "__getitem__", counted_read):
                        checked = verify(movie, references, rows, captions, ffmpeg)
                    self.assertEqual(reads, {"picture": 1, "caption": 1, "no_caption": 1, "audio": 1})
                    self.assertEqual(checked["frames"], 15)
                    self.assertEqual(checked["voice_start_after_action_frames"], 2)


if __name__ == "__main__":
    unittest.main()
