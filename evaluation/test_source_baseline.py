import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from source_baseline import load_partition
from evaluate_predictions import score_rows

class ProtocolTests(unittest.TestCase):
    def test_fixed_labels_and_false_positive_rate(self):
        result = score_rows([{'true_label':'legitimate','predicted_label':'fraud'}], ['fraud','legitimate','absent'])
        self.assertEqual(result['per_class']['absent']['support'], 0)
        self.assertEqual(result['legitimate_false_positive_rate'], 1)
        with self.assertRaises(ValueError):
            score_rows([{'true_label':'wrong','predicted_label':'fraud'}], ['fraud','legitimate'])

    def test_source_split_checks_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root/'data.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['text','binary_label']);writer.writeheader()
                for text,label in [('train0','0'),('train1','1'),('test','1')]:writer.writerow({'text':text,'binary_label':label})
            rows = [dict(row_index=i, sample_id=hashlib.sha256(text.encode()).hexdigest(),dataset=source) for i,(text,source) in enumerate([('train0','other'),('train1','other'),('test','difraud_sms')])]
            (root/'provenance.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            train,test=load_partition(root/'data.csv',root/'provenance.jsonl','difraud_')
            self.assertEqual((len(train),len(test)),(2,1))
            rows[0]['sample_id']='wrong'
            (root/'provenance.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):load_partition(root/'data.csv',root/'provenance.jsonl','difraud_')

if __name__ == '__main__':unittest.main()
