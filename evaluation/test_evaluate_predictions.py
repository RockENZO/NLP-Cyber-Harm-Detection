import unittest
import tempfile
from pathlib import Path

from evaluate_predictions import score_rows, main


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

    def test_ambiguous_csv_headers_rejected_before_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            labels=root/'labels.json'
            labels.write_text('["fraud","legitimate"]')
            output=root/'report.json'
            for header, record in [
                ('true_label,true_label,predicted_label', 'legitimate,fraud,fraud'),
                ('true_label,predicted_label,predicted_label', 'fraud,legitimate,fraud'),
                ('true_label,predicted_label,', 'fraud,fraud,x'),
            ]:
                predictions=root/'predictions.csv'
                predictions.write_text(header+'\n'+record+'\n')
                with self.subTest(header=header), self.assertRaises(SystemExit):
                    main([str(predictions),'--labels',str(labels),'--output',str(output)])
                self.assertFalse(output.exists())

    def test_empty_rows_rejected(self):
        with self.assertRaises(ValueError):
            score_rows([])


if __name__ == "__main__":
    unittest.main()
