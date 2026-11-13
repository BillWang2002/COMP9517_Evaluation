"""Strict cache validation and shared COCO/micro diagnostic reporting."""
import csv
import json
from pathlib import Path

from bird_project.evaluate import annotation_fingerprint,calculate_metrics,diagnostic_metrics,write_json,validate_records
from .settings import CAP,FUSION_PARAMETERS,MODELS,ROOT,SPLITS,WORKING_SCORES


def output_directory(path):
    path=Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output must be a new or empty directory: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate_cache(cache,allow_partial=False):
    split=cache.get('split');model=cache.get('model_id')
    if split not in SPLITS or model not in WORKING_SCORES:raise ValueError('Unknown split/model in cache.')
    if split=='test' and model=='FUSION':raise ValueError('Fusion is validation-only.')
    records=cache['records'];validate_records(records)
    fingerprint=annotation_fingerprint(records)
    if fingerprint!=cache.get('annotation_fingerprint'):raise ValueError('Cache annotation fingerprint mismatch.')
    with (ROOT/'data/splits.csv').open(newline='') as f:
        expected={r['file_name'] for r in csv.DictReader(f) if r['split']==split}
    actual={r['file_name'] for r in records};partial=actual!=expected
    if cache.get('partial') is not partial:raise ValueError('Incorrect partial-cache flag.')
    if partial and (split!='val' or not allow_partial or not actual<=expected):raise ValueError('Incomplete/foreign split; partial val requires --allow-partial.')
    if not partial:
        if fingerprint!=SPLITS[split]['fingerprint']:raise ValueError('Cache differs from official supplied GT.')
    names=('A1','B0') if model=='FUSION' else (model,)
    if cache.get('checkpoint_hashes')!={m:MODELS[m]['sha256'] for m in names}:raise ValueError('Cache checkpoint provenance differs.')
    if cache.get('model_settings')!={m:MODELS[m] for m in names}:raise ValueError('Cache model settings differ.')
    env=cache.get('environment',{})
    if env.get('input_size')!=640 or env.get('dtype')!='float32' or env.get('batch_size')!=1:raise ValueError('Cache inference protocol differs.')
    if cache.get('postprocessing')!=(FUSION_PARAMETERS if model=='FUSION' else None):raise ValueError('Postprocessing parameters differ.')
    return cache


def load_cache(path,allow_test=False,allow_partial=False):
    cache=json.loads(Path(path).read_text())
    if cache.get('split')=='test' and not allow_test:raise ValueError('Reading test metrics requires --allow-test.')
    return validate_cache(cache,allow_partial)


def evaluate(cache,allow_partial=False):
    validate_cache(cache,allow_partial)
    result=calculate_metrics(cache['records'],primary_max_detections=100,max_detections=CAP,
                             score_threshold=.5,diagnostic_iou_threshold=.5)
    name=cache['model_id']
    result.update(model_id=name,split=cache['split'],partial=cache['partial'],
                  operating_point_diagnostics=diagnostic_metrics(cache['records'],WORKING_SCORES[name],.5,CAP),
                  checkpoint_hashes=cache['checkpoint_hashes'],timing=cache.get('timing'),
                  note='AP uses all cached scores; common score0.5 and fixed model-specific working points are separate. TN is undefined.')
    return result


def comparison_row(result):
    ap=result['metrics_by_max_detections']['100'];w=result['operating_point_diagnostics'];d=result['diagnostics']
    return {'model':result['model_id'],'split':result['split'],'partial':result['partial'],'images':result['image_count'],
            'GT':d['ground_truth_count'],'AP50':ap['map_50'],'AP50_95':ap['map'],'AP75':ap['map_75'],
            'small_AP50_95':ap['map_small'],'AP50_cap300':result['metrics_by_max_detections']['300']['map_50'],
            'common_P':d['precision'],'common_R':d['recall'],'common_F1':d['f1'],
            'working_score':w['score_threshold'],'working_TP':w['tp'],'working_FP':w['fp'],'working_FN':w['fn'],
            'working_P':w['precision'],'working_R':w['recall'],'working_F1':w['f1']}


def save_results(out,results):
    if len({(m['split'],m['annotation_fingerprint'],m['partial']) for m in results})!=1:raise ValueError('Comparison requires identical images/GT.')
    rows=[comparison_row(m) for m in results]
    for result in results:write_json(out/f"{result['model_id']}_metrics.json",result)
    write_csv(out/'comparison.csv',rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))
