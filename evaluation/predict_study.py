"""Predict with a locally trained nine-class study artifact and its frozen selection."""
import argparse
import json
from pathlib import Path
from performance_study import shifted_predict
from source_baseline import digest_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/nine-class-study/models'))
    parser.add_argument('--text',required=True)
    args=parser.parse_args()
    selection=json.loads((args.run/'selection.json').read_text())['chosen']
    artifact=args.run/(selection['candidate']+'.joblib')
    if digest_file(artifact)!=selection['model_sha256']:parser.error('Model artifact does not match reviewed selection hash')
    # joblib is for trusted locally generated artifacts; never load an untrusted download.
    import joblib
    pipeline=joblib.load(artifact)
    scores=pipeline.decision_function([args.text])
    label=shifted_predict(scores,pipeline.classes_,selection['legitimate_margin_offset'])[0]
    print(json.dumps({'label':str(label),'model':selection['candidate'],
                     'legitimate_margin_offset':selection['legitimate_margin_offset'],
                     'note':'Decision margins are not calibrated probabilities; this research classifier can make mistakes.'},indent=2))

if __name__=='__main__':main()
