import unittest

from evaluate_predictions import score_rows


class ScoreRowsTests(unittest.TestCase):
    def test_per_class_and_macro_metrics(self):
        rows = [
            {"true_label": "fraud", "predicted_label": "fraud"},
            {"true_label": "fraud", "predicted_label": "legitimate"},
            {"true_label": "legitimate", "predicted_label": "legitimate"},
        ]
        result = score_rows(rows)
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["fraud"]["recall"], 0.5)
        self.assertEqual(result["confusion_matrix"]["fraud"]["legitimate"], 1)

    def test_empty_rows_rejected(self):
        with self.assertRaises(ValueError):
            score_rows([])


if __name__ == "__main__":
    unittest.main()
