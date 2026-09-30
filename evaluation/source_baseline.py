"""Train an auditable binary TF-IDF baseline with a source-family holdout."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_partition(data_path, provenance_path, holdout_prefix):
    train, test = [], []
    seen = set()
    with Path(data_path).open(newline='') as data, Path(provenance_path).open() as provenance:
        for row, raw in itertools.zip_longest(csv.DictReader(data), provenance):
            if row is None or raw is None:
                raise ValueError('Data/provenance row counts differ')
            source = json.loads(raw)
            identity = hashlib.sha256(row['text'].encode()).hexdigest()
            if source['sample_id'] != identity or source['row_index'] != len(seen):
                raise ValueError('Provenance does not match dataset row order/text')
            if identity in seen:
                raise ValueError('Duplicate text across input records')
            seen.add(identity)
            if row['binary_label'] not in ('0', '1'):
                raise ValueError('Invalid binary label')
            record = (identity, row['text'], int(row['binary_label']), source['dataset'])
            (test if source['dataset'].startswith(holdout_prefix) else train).append(record)
    if not train or not test or len({r[2] for r in train}) != 2:
        raise ValueError('Need non-empty train/test and both training labels')
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--holdout-prefix', default='difraud_')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/source-baseline'))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import sklearn
    import numpy
    import scipy
    from evaluate_predictions import score_rows
    train, test = load_partition(args.data, args.provenance, args.holdout_prefix)
    model = Pipeline([('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
                      ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear'))])
    model.fit([r[1] for r in train], [r[2] for r in train])
    predictions = model.predict([r[1] for r in test])
    rows = [{'sample_id': r[0], 'source': r[3], 'true_label': 'fraud' if r[2] else 'legitimate',
             'predicted_label': 'fraud' if int(p) else 'legitimate'} for r,p in zip(test,predictions)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n');writer.writeheader();writer.writerows(rows)
    split = {'train_ids': sorted(r[0] for r in train), 'test_ids': sorted(r[0] for r in test), 'holdout_prefix': args.holdout_prefix}
    (args.output_dir/'split_manifest.json').write_text(json.dumps(split, sort_keys=True)+'\n')
    report = score_rows(rows, labels=['fraud', 'legitimate'])
    report.update({'task': 'binary source-family holdout baseline', 'model': 'TF-IDF + balanced LogisticRegression',
        'train_samples': len(train), 'holdout_prefix': args.holdout_prefix,
        'train_sources': dict(sorted(Counter(r[3] for r in train).items())),
        'test_sources': dict(sorted(Counter(r[3] for r in test).items())),
        'source_metrics': {source: score_rows([r for r in rows if r['source']==source], labels=['fraud', 'legitimate']) for source in sorted({r['source'] for r in rows})},
        'input_sha256': digest_file(args.data), 'provenance_sha256': digest_file(args.provenance),
        'split_manifest_sha256': digest_file(args.output_dir/'split_manifest.json'),
        'predictions_sha256': digest_file(args.output_dir/'predictions.csv'),
        'parameters': {'max_features':40000,'ngram_range':[1,2],'min_df':2,'sublinear_tf':True,'C':1.0,'max_iter':1000,'random_state':42,'class_weight':'balanced','solver':'liblinear'},
        'environment': {'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__},
        'limitations': ['Source grouping is not proof of semantic or template independence.',
            'This baseline is newly trained with excluded sources; it does not evaluate bundled BERT/T5/BART/GPT checkpoints.',
            'Upstream inherited labels and synthetic records require separate validation.',
            'Binary fraud labels include spam under the corpus definition.']})
    (args.output_dir/'metrics.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ['train_samples','samples','accuracy','macro_f1','legitimate_false_positive_rate']},indent=2))

if __name__ == '__main__':
    main()
