import unittest
import contextlib
import io
import os
import tempfile
from unittest.mock import patch
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

    def test_report_cannot_overwrite_evaluation_inputs_or_aliases(self):
        for alias in ['predictions','labels','symlink','hardlink']:
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);predictions=root/'predictions.csv';labels=root/'labels.json'
                predictions.write_text('true_label,predicted_label\nfraud,fraud\n')
                labels.write_text('["fraud","legitimate"]')
                before=(predictions.read_bytes(),labels.read_bytes())
                if alias=='predictions':output=predictions
                elif alias=='labels':output=labels
                else:
                    output=root/'alias.json'
                    if alias=='symlink':output.symlink_to(predictions)
                    else:os.link(predictions,output)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(predictions),'--labels',str(labels),'--output',str(output)])
                self.assertEqual((predictions.read_bytes(),labels.read_bytes()),before)

    def test_alias_created_after_validation_cannot_destroy_input(self):
        for target_name in ['predictions','labels']:
            for alias in ['symlink','hardlink']:
                with self.subTest(target=target_name,alias=alias), tempfile.TemporaryDirectory() as folder:
                    root=Path(folder);predictions=root/'predictions.csv';labels=root/'labels.json';output=root/'report.json'
                    predictions.write_text('true_label,predicted_label\nfraud,fraud\n')
                    labels.write_text('["fraud","legitimate"]')
                    before=(predictions.read_bytes(),labels.read_bytes())
                    target=predictions if target_name=='predictions' else labels
                    def score_then_alias(*args,**kwargs):
                        result=score_rows(*args,**kwargs)
                        if alias=='symlink':output.symlink_to(target)
                        else:os.link(target,output)
                        return result
                    with patch('evaluate_predictions.score_rows',side_effect=score_then_alias), contextlib.redirect_stdout(io.StringIO()):
                        main([str(predictions),'--labels',str(labels),'--output',str(output)])
                    self.assertEqual((predictions.read_bytes(),labels.read_bytes()),before)
                    import json
                    self.assertEqual(json.loads(output.read_text())['accuracy'],1.0)

    def test_failed_replace_preserves_previous_report_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);predictions=root/'predictions.csv';labels=root/'labels.json';output=root/'report.json'
            predictions.write_text('true_label,predicted_label\nfraud,fraud\n')
            labels.write_text('["fraud","legitimate"]');output.write_text('previous report')
            with patch('evaluate_predictions.Path.replace',side_effect=OSError('write failure')), self.assertRaises(OSError):
                main([str(predictions),'--labels',str(labels),'--output',str(output)])
            self.assertEqual(output.read_text(),'previous report')
            self.assertEqual(list(root.glob('.evaluation-*')),[])

    def test_report_can_be_written_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);predictions=root/'predictions.csv';labels=root/'labels.json';output=root/'reports/eval.json'
            predictions.write_text('true_label,predicted_label\nfraud,fraud\n')
            labels.write_text('["fraud","legitimate"]')
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(predictions),'--labels',str(labels),'--output',str(output)])
            import json
            self.assertEqual(json.loads(output.read_text())['accuracy'],1.0)

    def test_hash_identifies_bytes_actually_scored_when_source_changes(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / 'predictions.csv'
            labels = root / 'labels.json'
            output = root / 'report.json'
            original = b'true_label,predicted_label\r\nfraud,fraud\r\n'
            predictions.write_bytes(original)
            labels.write_text('["fraud","legitimate"]')
            def score_then_replace(*args, **kwargs):
                result = score_rows(*args, **kwargs)
                predictions.write_text('true_label,predicted_label\nfraud,legitimate\n')
                return result
            with patch('evaluate_predictions.score_rows', side_effect=score_then_replace), contextlib.redirect_stdout(io.StringIO()):
                main([str(predictions), '--labels', str(labels), '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report['accuracy'], 1.0)
            self.assertEqual(report['predictions_sha256'], hashlib.sha256(original).hexdigest())
            self.assertNotEqual(report['predictions_sha256'], hashlib.sha256(predictions.read_bytes()).hexdigest())

    def test_ragged_or_malformed_csv_cannot_publish_a_report(self):
        for record in ['fraud,fraud,extra', 'fraud', '"fraud,fraud', '"fraud"oops,fraud']:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                predictions, labels, output = root/'p.csv', root/'l.json', root/'report.json'
                predictions.write_text('true_label,predicted_label\n' + record + '\n')
                labels.write_text('["fraud","legitimate"]')
                output.write_text('previous verified report')
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(predictions), '--labels', str(labels), '--output', str(output)])
                self.assertEqual(output.read_text(), 'previous verified report')
                self.assertEqual(list(root.glob('.evaluation-*')), [])

    def test_metadata_quote_validation_preserves_valid_multiline_csv(self):
        import csv
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions, labels, output = root/'p.csv', root/'l.json', root/'report.json'
            labels.write_text('["fraud","legitimate"]')
            predictions.write_text('true_label,predicted_label,note\nfraud,fraud,bad"quote\n')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(predictions), '--labels', str(labels), '--output', str(output)])
            self.assertFalse(output.exists())
            with predictions.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['true_label', 'predicted_label', 'note'])
                writer.writerow(['fraud', 'fraud', 'line 1, "quoted"\nline 2'])
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(predictions), '--labels', str(labels), '--output', str(output)])
            self.assertEqual(json.loads(output.read_text())['accuracy'], 1.0)

    def test_label_universe_requires_an_ordered_array_of_nonempty_strings(self):
        import json
        for universe in ['fraud', {'fraud': 1}, ['fraud', ''], ['fraud', '   '], ['fraud', 1], ['fraud', ['nested']], ['fraud', 'fraud']]:
            with self.subTest(labels=universe), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                predictions, labels, output = root/'p.csv', root/'l.json', root/'report.json'
                predictions.write_text('true_label,predicted_label\nfraud,fraud\n')
                labels.write_text(json.dumps(universe))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(predictions), '--labels', str(labels), '--output', str(output)])
                self.assertFalse(output.exists())

    def test_label_hash_uses_the_snapshot_that_defined_macro_f1(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions, labels, output = root/'p.csv', root/'l.json', root/'report.json'
            predictions.write_text('sample_id,true_label,predicted_label\nitem-1,fraud,fraud\n')
            original = b'["fraud", "legitimate"]\r\n'
            labels.write_bytes(original)
            def score_then_replace(*args, **kwargs):
                result = score_rows(*args, **kwargs)
                labels.write_text('["fraud"]')
                return result
            with patch('evaluate_predictions.score_rows', side_effect=score_then_replace), contextlib.redirect_stdout(io.StringIO()):
                main([str(predictions), '--labels', str(labels), '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report['macro_f1'], 0.5)
            self.assertEqual(report['labels_sha256'], hashlib.sha256(original).hexdigest())

    def test_single_pass_iterator_preserves_explicit_and_inferred_metrics(self):
        records = [
            {"true_label": "fraud", "predicted_label": "fraud"},
            {"true_label": "legitimate", "predicted_label": "fraud"},
            {"true_label": "legitimate", "predicted_label": "legitimate"},
        ]
        for labels in [None, ["legitimate", "fraud", "absent"]]:
            with self.subTest(labels=labels):
                stream = (row for row in records)
                self.assertEqual(score_rows(stream, labels), score_rows(records, labels))
                self.assertEqual(list(stream), [])

    def test_late_invalid_label_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions, labels, output = root/'p.csv', root/'l.json', root/'report.json'
            predictions.write_text('true_label,predicted_label\n' + 'fraud,fraud\n' * 1000 + 'fraud,   \n')
            labels.write_text('["fraud","legitimate"]')
            output.write_text('previous verified report')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(predictions), '--labels', str(labels), '--output', str(output)])
            self.assertEqual(output.read_text(), 'previous verified report')
            self.assertEqual(list(root.glob('.evaluation-*')), [])

    def test_empty_rows_rejected(self):
        with self.assertRaises(ValueError):
            score_rows([])


class BenchmarkContractTests(unittest.TestCase):
    def test_small_synthetic_measurement_preserves_previous_results(self):
        import json
        from benchmark_scoring import main as measure
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'measurement.json'
            with contextlib.redirect_stdout(io.StringIO()):
                measure(['--rows', '20', '--output', str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result['rows'], 20)
            self.assertEqual(len(result['report_sha256']), 64)
            before = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                measure(['--rows', '20', '--output', str(output)])
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_benchmark_size_creates_no_measurement(self):
        from benchmark_scoring import main as measure
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'measurement.json'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                measure(['--rows', '0', '--output', str(output)])
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
