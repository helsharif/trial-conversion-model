import unittest
from datetime import date

from trial_conversion_model.dashboard_data import (
    Artifact,
    choose_verdicts,
    list_artifacts,
    parse_scores,
    parse_verdict,
)


class DashboardTests(unittest.TestCase):
    def test_ranking_and_identifiers(self):
        frame = parse_scores(b'trial_id,conversion_probability\n001,0.9\n002,0.1\n003,0.1\n')
        self.assertEqual(frame.trial_id.tolist(), ['002', '003', '001'])
        self.assertEqual(frame['rank'].tolist(), [1, 2, 3])
        self.assertAlmostEqual(frame.cancellation_risk.iloc[0], 0.9)

    def test_invalid_probabilities_are_not_ranked(self):
        for value in ['-0.1', '1.1', '', 'oops']:
            with self.assertRaises(ValueError):
                parse_scores(f'trial_id,conversion_probability\n1,{value}\n'.encode())

    def test_invalid_verdict_is_not_healthy(self):
        for raw in [b'{}', b'{"drift":"false","drifted_share":0}', b'{"drift":false,"drifted_share":2}']:
            with self.assertRaises(ValueError):
                parse_verdict(raw)

    def test_measured_verdict_beats_newer_backfill(self):
        day = date(2026, 10, 1)
        measured = Artifact('monitoring/cohort_2026-10-01.json', day, '2026-10-01', 'a')
        backfill = Artifact('monitoring/drift_cohort_2026-10-01.json', day, '2026-10-17', 'b')
        selected = choose_verdicts([(measured, {'drift': True}), (backfill, {'drift': False, 'source': 'backfill'})])
        self.assertTrue(selected[day][1]['drift'])
        self.assertEqual(selected[day][0], measured)

    def test_inventory_paginates_and_uses_artifact_dates(self):
        class Client:
            def get_paginator(self, operation):
                return self

            def paginate(self, **kwargs):
                for day in ['2026-10-17', '2026-09-17']:
                    yield {'Contents': [{'Key': f'predictions/scored_{day}.csv', 'Size': 10,
                                         'LastModified': date(2026, 10, 17), 'ETag': day}]}
        result = list_artifacts(Client(), 'bucket', 'predictions/', '.csv')
        self.assertEqual([a.day.isoformat() for a in result], ['2026-09-17', '2026-10-17'])


if __name__ == '__main__':
    unittest.main()
