import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from source_baseline import load_partition


class SourceSnapshotTests(unittest.TestCase):
    def test_report_hashes_identify_parsed_bytes_even_if_paths_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root/'data.csv'; provenance = root/'provenance.jsonl'
            stream = io.StringIO(newline=''); writer = csv.writer(stream); writer.writerow(['text', 'binary_label'])
            rows = [('train0','0','other'),('train1','1','other'),('test','1','difraud_sms')]
            writer.writerows((text,label) for text,label,_ in rows); data.write_bytes(stream.getvalue().encode())
            provenance.write_text(''.join(json.dumps({'row_index':i,'sample_id':hashlib.sha256(text.encode()).hexdigest(),'dataset':source})+'\n' for i,(text,_,source) in enumerate(rows)))
            originals = {p: p.read_bytes() for p in [data, provenance]}
            original_read = Path.read_bytes
            def mutate_after_read(path):
                value = original_read(path)
                if path in originals: path.write_bytes(b'changed while training')
                return value
            hashes = {}
            with patch.object(Path, 'read_bytes', mutate_after_read):
                train, test = load_partition(data, provenance, 'difraud_', input_hashes=hashes)
            self.assertEqual((len(train),len(test)), (2,1))
            self.assertEqual(hashes, {'input_sha256':hashlib.sha256(originals[data]).hexdigest(),
                                     'provenance_sha256':hashlib.sha256(originals[provenance]).hexdigest()})
