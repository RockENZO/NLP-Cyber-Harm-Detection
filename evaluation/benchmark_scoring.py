"""Measure scorer allocations on synthetic labels, never model performance.

The optional --scorer path is imported as Python code. Only use trusted source
(e.g. the reviewed baseline), not a downloaded untrusted module.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import tempfile
import time
import tracemalloc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer', type=Path, default=Path(__file__).with_name('evaluate_predictions.py'))
    parser.add_argument('--rows', type=int, default=150_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.rows <= 1_000_000:
        parser.error('--rows must be between 1 and 1000000')
    if args.output.exists() or args.output.is_symlink():
        parser.error('Choose a new output file; previous measurements are preserved')
    spec = importlib.util.spec_from_file_location('measured_scorer', args.scorer)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions, labels, report = root/'predictions.csv', root/'labels.json', root/'report.json'
        labels.write_bytes(b'["legitimate","fraud","absent"]')
        with predictions.open('w', encoding='utf-8', newline='') as stream:
            stream.write('true_label,predicted_label,sample_id\n')
            for index in range(args.rows):
                actual = 'legitimate' if index % 2 == 0 else 'fraud'
                predicted = actual if index % 5 else ('fraud' if actual == 'legitimate' else 'legitimate')
                stream.write(f'{actual},{predicted},{index}\n')
        tracemalloc.start()
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                scorer.main([str(predictions), '--labels', str(labels), '--output', str(report)])
            seconds = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        result = {'scope': 'Synthetic scorer allocation workload, not trained-model evaluation',
                  'python': platform.python_version(), 'platform': platform.platform(),
                  'rows': args.rows, 'seconds': seconds, 'tracemalloc_peak_bytes': peak,
                  'scorer_sha256': hashlib.sha256(args.scorer.read_bytes()).hexdigest(),
                  'input_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
                  'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest(),
                  'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
                  'limitations': 'tracemalloc covers Python allocations, not full process RSS; timing includes its overhead and is not production throughput. Input byte/text snapshots are retained.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
