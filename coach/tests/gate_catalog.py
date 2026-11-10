"""Completion gate for the two analysis units. Run it by name, discover does not pick it up.

    python -m unittest coach.tests.gate_catalog

It stays red until both units have written a metric spec for every finding they own.
"""
import unittest

from coach import schema as S
from coach.catalog import METRIC_CATALOG
from coach.catalog_body import BODY_SPECS
from coach.catalog_move import MOVE_SPECS

BODY = {S.Category.TECHNIQUE, S.Category.CONTACT}
MOVE = {S.Category.FOOTWORK, S.Category.TACTICS}


def owned(specs, categories):
    return {s.finding_id for s in specs} & {fid for fid, (c, _) in S.FINDING_CATALOG.items() if c in categories}


class CatalogComplete(unittest.TestCase):
    def test_every_finding_id_has_a_metric_spec(self):
        covered = {s.finding_id for s in METRIC_CATALOG.values()}
        self.assertEqual(sorted(set(S.FINDING_CATALOG) - covered), [])

    def test_each_unit_specs_only_its_own_categories(self):
        wrong = [s.name for s in BODY_SPECS if S.FINDING_CATALOG[s.finding_id][0] not in BODY]
        wrong += [s.name for s in MOVE_SPECS if S.FINDING_CATALOG[s.finding_id][0] not in MOVE]
        self.assertEqual(wrong, [])

    def test_each_unit_covers_all_of_its_findings(self):
        body_ids = {fid for fid, (c, _) in S.FINDING_CATALOG.items() if c in BODY}
        move_ids = {fid for fid, (c, _) in S.FINDING_CATALOG.items() if c in MOVE}
        self.assertEqual(owned(BODY_SPECS, BODY), body_ids)
        self.assertEqual(owned(MOVE_SPECS, MOVE), move_ids)


if __name__ == "__main__":
    unittest.main()
