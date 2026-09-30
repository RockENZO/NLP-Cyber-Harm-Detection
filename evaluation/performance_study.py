"""Fresh nine-class study: explicit split, template grouping, validation-only selection."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata
from source_baseline import digest_file
from evaluate_predictions import score_rows

LABELS=json.loads(Path(__file__).with_name('labels.json').read_text())
RULE='Unicode NFKC lowercase; replace URLs/emails/numbers; collapse whitespace; SHA256; preserve official DIFrauD boundaries at group level'


def template_id(text):
    normalized=unicodedata.normalize('NFKC',text).lower()
    normalized=re.sub(r'https?://\S+|www\.\S+',' <url> ',normalized)
    normalized=re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}',' <email> ',normalized)
    normalized=re.sub(r'\d+',' <number> ',normalized)
    normalized=' '.join(normalized.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def source_role(source):
    path=source['source_file']
    if source['dataset'].startswith('difraud_'):
        if path.endswith('/test.jsonl'):return 'test'
        if path.endswith('/validation.jsonl'):return 'val'
        if path.endswith('/train.jsonl'):return 'train'
        raise ValueError('Unknown official source split')
    return None


def prepare(data,provenance,output):
    if output.exists():raise ValueError('Output already exists')
    groups=defaultdict(list);seen=set()
    with data.open(newline='') as stream,provenance.open() as origins:
        for row,line in itertools.zip_longest(csv.DictReader(stream),origins):
            if row is None or line is None:raise ValueError('Provenance row counts differ')
            source=json.loads(line);identity=hashlib.sha256(row['text'].encode()).hexdigest()
            if identity!=source['sample_id'] or source['row_index']!=len(seen) or identity in seen:raise ValueError('Invalid provenance/text identity')
            seen.add(identity)
            if row['detailed_category'] not in LABELS:raise ValueError('Unknown category')
            row.update(sample_id=identity,group_id=template_id(row['text']),source=source['dataset'],source_file=source['source_file'],official_role=source_role(source))
            groups[row['group_id']].append(row)
    splits={role:[] for role in ['train','val','test']};excluded=Counter();promotion=0
    for group,rows in sorted(groups.items()):
        if len({r['detailed_category'] for r in rows})!=1:
            excluded['conflicting_template_labels']+=len(rows);continue
        roles={r['official_role'] for r in rows if r['official_role']}
        role='test' if 'test' in roles else 'val' if 'val' in roles else 'train' if roles else None
        if role is None:
            bucket=int(hashlib.sha256(('20260930:'+group).encode()).hexdigest()[:8],16)%100
            role='test' if bucket<10 else 'val' if bucket<20 else 'train'
        eligible=[r for r in rows if r['official_role']==role] or rows
        representative=min(eligible,key=lambda r:r['sample_id'])
        splits[role].append(representative)
        excluded['normalized_duplicates']+=len(rows)-1
        promotion+=len(roles)>1
    output.mkdir(parents=True)
    manifest={'protocol':'newly fitted nine-class in-corpus comparison; official DIFrauD test/validation boundaries retained after template grouping',
              'template_rule':RULE,'seed':20260930,'input_sha256':digest_file(data),'provenance_sha256':digest_file(provenance),
              'input_rows':len(seen),'groups':len(groups),'excluded':dict(excluded),'official_group_promotions':promotion,
              'counts':{role:len(rows) for role,rows in splits.items()},
              'class_support':{role:dict(Counter(r['detailed_category'] for r in rows)) for role,rows in splits.items()},
              'split_ids':{role:[{'sample_id':r['sample_id'],'group_id':r['group_id'],'source':r['source']} for r in rows] for role,rows in splits.items()},
              'limitations':['This is an in-corpus benchmark, not unseen-source validation.',
                  'The historical models may have trained on these records; this study fits new models from scratch.',
                  'Normalization catches some templates, not every semantic near duplicate.',
                  'Inherited source labels, synthetic records and source-specific shortcuts limit external generalization.']}
    for role,rows in splits.items():
        if set(r['detailed_category'] for r in rows)!=set(LABELS):raise ValueError('Every split must support all nine classes')
        with (output/(role+'.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return {k:v for k,v in manifest.items() if k!='split_ids'}


def read_split(study,role):
    return [json.loads(line) for line in (study/(role+'.jsonl')).open()]


def shifted_predict(scores,classes,offset):
    import numpy as np
    values=scores.copy();values[:,list(classes).index('legitimate')]+=offset
    return np.asarray(classes)[values.argmax(axis=1)]


def metrics(rows,predictions):
    nine=score_rows([{'true_label':r['detailed_category'],'predicted_label':str(p)} for r,p in zip(rows,predictions)],labels=LABELS)
    binary=score_rows([{'true_label':'legitimate' if r['detailed_category']=='legitimate' else 'fraud','predicted_label':'legitimate' if p=='legitimate' else 'fraud'} for r,p in zip(rows,predictions)],labels=['fraud','legitimate'])
    return {'nine_class':nine,'binary':binary}


def select(study,run):
    if run.exists():raise ValueError('Run exists')
    import joblib,numpy as np,platform,sklearn,time
    from sklearn.pipeline import Pipeline,FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from threadpoolctl import threadpool_limits
    train=read_split(study,'train');val=read_split(study,'val')
    texts=[r['text'] for r in train];targets=[r['detailed_category'] for r in train]
    candidates=[('word_baseline',False,None,1.0),('word_char_balanced',True,'balanced',1.0)]
    run.mkdir(parents=True);leaderboard=[]
    with threadpool_limits(limits=2):
        for name,hybrid,weight,c in candidates:
            start=time.monotonic()
            word=TfidfVectorizer(max_features=50000,ngram_range=(1,2),min_df=2,sublinear_tf=True,dtype=np.float32)
            features=FeatureUnion([('word',word),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=70000,sublinear_tf=True,dtype=np.float32))]) if hybrid else word
            pipeline=Pipeline([('features',features),('classifier',LinearSVC(C=c,class_weight=weight,dual='auto',random_state=42,max_iter=5000))])
            pipeline.fit(texts,targets);scores=pipeline.decision_function([r['text'] for r in val]);classes=pipeline.classes_
            modelpath=run/(name+'.joblib');joblib.dump(pipeline,modelpath,compress=3)
            for offset in ([0] if not hybrid else [0,.25,.5,.75,1]):
                report=metrics(val,shifted_predict(scores,classes,offset))
                leaderboard.append({'candidate':name,'legitimate_margin_offset':offset,'validation':report,'model_sha256':digest_file(modelpath),'elapsed_seconds':time.monotonic()-start})
            print(name,'validation candidates',json.dumps([{k:r[k] for k in ['candidate','legitimate_margin_offset','validation']} for r in leaderboard if r['candidate']==name]),flush=True)
    # Predeclared rule: validation FPR <=3%, binary fraud recall >=80%; maximize nine-class macro F1.
    eligible=[r for r in leaderboard if r['validation']['binary']['legitimate_false_positive_rate']<=.03 and r['validation']['binary']['per_class']['fraud']['recall']>=.8]
    chosen=max(eligible or leaderboard,key=lambda r:r['validation']['nine_class']['macro_f1'])
    selection={'rule':'validation FPR<=0.03 and fraud recall>=0.80, then maximum nine-class macro F1; if none eligible report unmet gate',
               'gate_satisfied':bool(eligible),'chosen':chosen,'candidates':leaderboard,'split_manifest_sha256':digest_file(study/'manifest.json'),
               'runtime':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},
               'parameters':{'word_max_features':50000,'char_max_features':70000,'char_ngrams':[3,5],'C':1.0,'random_state':42,'max_iter':5000},
               'final_test_evaluated':False}
    (run/'selection.json').write_text(json.dumps(selection,indent=2,sort_keys=True)+'\n')
    print('SELECTED',chosen['candidate'],chosen['legitimate_margin_offset'],bool(eligible),flush=True)


def evaluate(study,run):
    import joblib
    if (run/'final_test.json').exists():raise ValueError('Final test already evaluated; preserve frozen results')
    selection=json.loads((run/'selection.json').read_text())
    if digest_file(study/'manifest.json')!=selection['split_manifest_sha256']:raise ValueError('Split changed')
    test=read_split(study,'test');reports={}
    chosen=selection['chosen']
    for name,offset in [('word_baseline',0),(chosen['candidate'],chosen['legitimate_margin_offset'])]:
        pipeline=joblib.load(run/(name+'.joblib'))
        predicted=shifted_predict(pipeline.decision_function([r['text'] for r in test]),pipeline.classes_,offset)
        report=metrics(test,predicted)
        report['by_source']={source:metrics([r for r in test if r['source']==source],[p for r,p in zip(test,predicted) if r['source']==source]) for source in sorted({r['source'] for r in test})}
        real_indices=[i for i,r in enumerate(test) if r['source'].startswith('difraud_') or r['source'].startswith('phishing_') or r['source'].startswith('spam_')]
        report['email_sms_jobs_subset']={'scope':'DIFrauD, phishing and spam source families; excludes popup and dialogue sources',
            'metrics':score_rows([{'true_label':test[i]['detailed_category'],'predicted_label':str(predicted[i])} for i in real_indices],labels=LABELS),
            'binary':score_rows([{'true_label':'legitimate' if test[i]['detailed_category']=='legitimate' else 'fraud',
                                'predicted_label':'legitimate' if predicted[i]=='legitimate' else 'fraud'} for i in real_indices],labels=['fraud','legitimate'])}
        # Keep the explicit nine-class prediction universe; absent classes have zero support.
        key='baseline' if not reports else 'selected'
        with (run/(key+'-predictions.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['sample_id','group_id','source','true_label','predicted_label'],lineterminator='\n');writer.writeheader()
            for row,pred in zip(test,predicted):writer.writerow({k:row[k] for k in ['sample_id','group_id','source']}|{'true_label':row['detailed_category'],'predicted_label':str(pred)})
        report.update(candidate=name,offset=offset,model_sha256=digest_file(run/(name+'.joblib')),predictions_sha256=digest_file(run/(key+'-predictions.csv')))
        reports[key]=report
    report={'protocol':json.loads((study/'manifest.json').read_text())['protocol'],'split_manifest_sha256':digest_file(study/'manifest.json'),
            'selection_sha256':digest_file(run/'selection.json'),'results':reports,'selection_rule':selection['rule'],
            'limitations':json.loads((study/'manifest.json').read_text())['limitations']}
    (run/'final_test.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:{'macro_f1':v['nine_class']['macro_f1'],'accuracy':v['nine_class']['accuracy'],'fpr':v['binary']['legitimate_false_positive_rate'],'fraud_recall':v['binary']['per_class']['fraud']['recall']} for k,v in reports.items()},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','select','evaluate'])
    parser.add_argument('--data',type=Path,default=Path('../data/artifacts/verified/final_fraud_detection_dataset.csv'))
    parser.add_argument('--provenance',type=Path,default=Path('../data/artifacts/verified/provenance.jsonl'))
    parser.add_argument('--study',type=Path,default=Path('runs/nine-class-study/split'))
    parser.add_argument('--run',type=Path,default=Path('runs/nine-class-study/models'))
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(args.data,args.provenance,args.study),indent=2))
    elif args.command=='select':select(args.study,args.run)
    else:evaluate(args.study,args.run)
