import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from performance_study import LABELS,prepare,template_id,source_role,read_split

class StudyTests(unittest.TestCase):
    def test_template_normalization(self):
        self.assertEqual(template_id('Pay 123 at https://a.example NOW'),template_id('pay 456 at https://b.example now'))
        self.assertEqual(template_id('CONTACT a@example.com'),template_id('contact b@example.org'))
        self.assertNotEqual(template_id('pay fee'),template_id('appointment confirmed'))

    def test_official_roles_and_disjoint_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data.csv';origins=root/'provenance.jsonl';rows=[];provenance=[]
            for label in LABELS:
                for role,filename in [('train','train'),('val','validation'),('test','test')]:
                    text=label+' '+role+' distinctive message'
                    rows.append(dict(text=text,detailed_category=label,binary_label='0' if label=='legitimate' else '1',data_type='email'))
                    provenance.append(dict(sample_id=hashlib.sha256(text.encode()).hexdigest(),row_index=len(provenance),dataset='difraud_fixture',source_file='difraud/fixture/'+filename+'.jsonl'))
            with data.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            origins.write_text(''.join(json.dumps(row)+'\n' for row in provenance))
            report=prepare(data,origins,root/'study')
            self.assertEqual(report['counts'],{'train':9,'val':9,'test':9})
            self.assertEqual(len(read_split(root/'study','train')),9)
            study_rows=[json.loads(line) for line in (root/'study/train.jsonl').read_text().splitlines()]
            study_rows[0]['text']='changed'
            (root/'study/train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in study_rows))
            with self.assertRaisesRegex(ValueError,'identities'):read_split(root/'study','train')
            manifest=json.loads((root/'study/manifest.json').read_text())
            ids={role:{row['group_id'] for row in records} for role,records in manifest['split_ids'].items()}
            for left,right in [('train','val'),('train','test'),('val','test')]:self.assertFalse(ids[left]&ids[right])
            with self.assertRaises(ValueError):prepare(data,origins,root/'study')
            provenance[0]['sample_id']='bad';origins.write_text(''.join(json.dumps(row)+'\n' for row in provenance))
            with self.assertRaisesRegex(ValueError,'identity'):prepare(data,origins,root/'bad')

    def test_unknown_official_split_rejected(self):
        with self.assertRaises(ValueError):source_role({'dataset':'difraud_sms','source_file':'other.csv'})

if __name__=='__main__':unittest.main()
