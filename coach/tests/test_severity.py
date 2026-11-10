import unittest

from coach import schema as S
from coach.catalog import build_catalog
from coach.severity import MetricSpec, check_findings, distance, score_for, severity_for


def spec(direction="lower_better", target=(0.0, 0.5), tolerance=0.1, finding_id="technique.stay_lower", name="hip_height_ratio"):
    return MetricSpec(name, "Hip height", "ratio", S.Basis.PROJECTED_2D, direction, target, tolerance, finding_id)


def finding_for(value, severity, score, confidence=0.8, n_obs=6):
    m = S.Metric("hip_height_ratio", "Hip height", value, "ratio", S.Basis.PROJECTED_2D, confidence, (0.0, 0.5), None, n_obs)
    moment = S.Moment(S.Role.NEAR, 2, 5, 8, "ready stance")
    return S.Finding(
        "technique.stay_lower", S.Category.TECHNIQUE, S.Role.NEAR, severity, "h", "e", (m,), (moment,), S.Tier.PREVIEW,
        fix_tags=("lower_stance",), score=score,
    )


class Distance(unittest.TestCase):
    def test_lower_better_only_counts_the_excess(self):
        s = spec("lower_better", (0.0, 0.5), 0.1)
        self.assertAlmostEqual(distance(s, 0.8), 3.0)
        self.assertEqual(distance(s, 0.2), 0.0)
        self.assertEqual(distance(s, 0.5), 0.0)

    def test_higher_better_only_counts_the_shortfall(self):
        s = spec("higher_better", (0.6, 1.0), 0.1)
        self.assertAlmostEqual(distance(s, 0.4), 2.0)
        self.assertEqual(distance(s, 0.9), 0.0)

    def test_in_range_counts_both_sides(self):
        s = spec("in_range", (0.4, 0.6), 0.1)
        self.assertAlmostEqual(distance(s, 0.2), 2.0)
        self.assertAlmostEqual(distance(s, 0.9), 3.0)
        self.assertEqual(distance(s, 0.5), 0.0)


class Severity(unittest.TestCase):
    def test_severity_never_falls_as_the_value_gets_worse(self):
        s = spec()
        ranks = [S.SEVERITY_RANK[severity_for(s, 0.5 + i * 0.01, 0.8, 6)] for i in range(0, 60)]
        self.assertEqual(ranks, sorted(ranks))

    def test_each_band_has_the_expected_severity(self):
        s = spec()
        got = [severity_for(s, v, 0.8, 6) for v in (0.4, 0.55, 0.65, 0.8)]
        self.assertEqual(got, [S.Severity.STRENGTH, S.Severity.INFO, S.Severity.IMPROVE, S.Severity.PRIORITY])

    def test_unreliable_metrics_never_rise_above_info(self):
        s = spec()
        self.assertEqual(severity_for(s, 0.9, 0.5, 6), S.Severity.INFO)
        self.assertEqual(severity_for(s, 0.9, 0.8, 2), S.Severity.INFO)
        self.assertEqual(severity_for(s, 0.4, 0.5, 6), S.Severity.INFO)
        self.assertEqual(severity_for(s, 0.4, 0.8, 2), S.Severity.INFO)

    def test_score_is_distance_times_confidence(self):
        self.assertAlmostEqual(score_for(spec(), 0.8, 0.5), 1.5)
        self.assertEqual(score_for(spec(), 0.3, 0.9), 0.0)


class SpecRules(unittest.TestCase):
    def test_bad_specs_are_rejected(self):
        for kw in (
            dict(finding_id="technique.made_up"),
            dict(direction="sideways"),
            dict(target=(0.6, 0.4)),
            dict(tolerance=0.0),
            dict(name="Hip Height"),
        ):
            with self.assertRaises(ValueError, msg=str(kw)):
                spec(**kw)

    def test_duplicate_names_are_rejected_when_building_the_catalog(self):
        with self.assertRaises(ValueError) as caught:
            build_catalog((spec(),), (spec(finding_id="technique.knee_bend"),))
        self.assertIn("specified twice", str(caught.exception))


class CheckFindings(unittest.TestCase):
    def test_a_finding_that_follows_the_rule_passes(self):
        check_findings([finding_for(0.8, S.Severity.PRIORITY, 2.4)], build_catalog((spec(),)))

    def test_a_hand_picked_severity_fails(self):
        with self.assertRaises(ValueError):
            check_findings([finding_for(0.8, S.Severity.IMPROVE, 2.4)], build_catalog((spec(),)))

    def test_a_finding_with_no_cataloged_metric_fails(self):
        with self.assertRaises(ValueError) as caught:
            check_findings([finding_for(0.8, S.Severity.PRIORITY, 2.4)], build_catalog((spec(finding_id="technique.knee_bend"),)))
        self.assertIn("no metric", str(caught.exception))

    def test_a_low_confidence_metric_cannot_carry_an_improve_finding(self):
        with self.assertRaises(ValueError):
            check_findings([finding_for(0.8, S.Severity.IMPROVE, 1.5, confidence=0.5)], build_catalog((spec(),)))


def absent_metric(name="hip_height_ratio"):
    return S.Metric(name, "Hip height", None, "ratio", S.Basis.PROJECTED_2D, 0.0, (0.0, 0.5), None, 0, None)


def unavailable_finding(severity=S.Severity.INFO, score=0.0, moments=(), metrics=None, fix_tags=()):
    return S.Finding(
        "technique.stay_lower", S.Category.TECHNIQUE, S.Role.NEAR, severity, "h", "e",
        metrics or (absent_metric(),), moments, S.Tier.PREVIEW, fix_tags=fix_tags, score=score,
    )


class UnavailableFindings(unittest.TestCase):
    def test_an_entirely_unavailable_finding_is_rejected_before_catalog_scoring(self):
        check_findings([unavailable_finding()], build_catalog((spec(),)))
        moment = S.Moment(S.Role.NEAR, 2, 5, 8, "ready stance")
        for make in (
            lambda: unavailable_finding(S.Severity.IMPROVE, fix_tags=("lower_stance",), moments=(moment,)),
            lambda: unavailable_finding(S.Severity.STRENGTH),
            lambda: unavailable_finding(score=0.0005),
            lambda: unavailable_finding(moments=(moment,)),
        ):
            with self.assertRaises(ValueError):
                make()

    def test_a_mixed_finding_is_scored_from_the_observed_metric_only(self):
        seen = S.Metric("ready_trunk_lean", "Lean", 0.8, "ratio", S.Basis.PROJECTED_2D, 0.8, (0.0, 0.5), None, 6)
        moment = S.Moment(S.Role.NEAR, 2, 5, 8, "ready stance")
        finding = unavailable_finding(
            S.Severity.PRIORITY, 2.4, (moment,), (absent_metric(), seen), ("lower_stance",),
        )
        catalog = build_catalog((spec(), spec(name="ready_trunk_lean", finding_id="technique.stay_lower")))
        check_findings([finding], catalog)
        quiet = unavailable_finding(metrics=(absent_metric(), seen))
        with self.assertRaises(ValueError):
            check_findings([quiet], catalog)

    def test_a_number_outside_the_catalog_does_not_make_a_finding_available(self):
        extra = S.Metric("not_in_catalog", "Extra", 0.9, "ratio", S.Basis.PROJECTED_2D, 0.9, None, None, 6)
        catalog = build_catalog((spec(),))
        check_findings([unavailable_finding(metrics=(absent_metric(), extra))], catalog)
        moment = S.Moment(S.Role.NEAR, 2, 5, 8, "ready stance")
        claimed = unavailable_finding(
            S.Severity.PRIORITY, 2.4, (moment,), (absent_metric(), extra), ("lower_stance",),
        )
        with self.assertRaises(ValueError) as caught:
            check_findings([claimed], catalog)
        self.assertIn("unavailable", str(caught.exception))
        tiny = unavailable_finding(score=0.0005, metrics=(absent_metric(), extra))
        with self.assertRaises(ValueError) as caught:
            check_findings([tiny], catalog)
        self.assertIn("score must be 0", str(caught.exception))

    def test_an_observed_zero_confidence_reading_stays_numeric(self):
        check_findings(
            [finding_for(0.8, S.Severity.INFO, 0.0, confidence=0.0)],
            build_catalog((spec(),)),
        )
        self.assertEqual(severity_for(spec(), 0.8, 0.0, 6), S.Severity.INFO)
        self.assertEqual(score_for(spec(), 0.8, 0.0), 0.0)

    def test_distance_stays_numeric(self):
        with self.assertRaises(TypeError):
            distance(spec(), None)


if __name__ == "__main__":
    unittest.main()
