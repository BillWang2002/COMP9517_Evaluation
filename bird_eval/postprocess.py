"""Fixed selective A1 supplementation (previous validation variant S15).

Only predicted boxes and scores determine acceptance. Ground truth is copied
for later evaluation and never enters the selection logic.
"""
from copy import deepcopy
from bird_project.evaluate import box_iou
from .settings import CAP,FUSION_PARAMETERS


def supplement(a,b):
    for key in ('file_name','width','height'):
        if key in a or key in b:
            if a.get(key)!=b.get(key):raise ValueError('A/B image metadata differ.')
    p=FUSION_PARAMETERS
    candidates=[(float(s),'B0',i,box,label) for i,(box,s,label) in enumerate(zip(b['boxes'],b['scores'],b['labels']))]
    support=[i for i,s in enumerate(b['scores']) if s>=p['b_support_min_score']]
    selected=[]
    for i in sorted(range(len(a['scores'])),key=lambda j:(-a['scores'][j],j)):
        box,score,label=a['boxes'][i],a['scores'][i],a['labels'][i]
        if score<p['a_min_score'] or (box[2]-box[0])*(box[3]-box[1])<p['a_min_original_box_area']:continue
        if any(label==b['labels'][j] and box_iou(box,b['boxes'][j])>=p['novelty_iou'] for j in support):continue
        if any(label==a['labels'][j] and box_iou(box,a['boxes'][j])>=p['a_supplement_nms_iou'] for j in selected):continue
        selected.append(i);candidates.append((score*p['a_score_scale'],'A1',i,box,label))
    candidates.sort(key=lambda x:(-x[0],x[1]!='B0',x[2]));candidates=candidates[:CAP]
    result={k:deepcopy(b[k]) for k in ('image_id','file_name','width','height','ground_truth') if k in b}
    result.update(boxes=[list(x[3]) for x in candidates],scores=[x[0] for x in candidates],labels=[x[4] for x in candidates],
                  prediction_sources=[{'model':x[1],'prediction_index':x[2]} for x in candidates])
    return result


def supplement_records(a_records,b_records):
    a={r['file_name']:r for r in a_records};b={r['file_name']:r for r in b_records}
    if len(a)!=len(a_records) or len(b)!=len(b_records) or a.keys()!=b.keys():raise ValueError('Unequal or duplicate A/B images.')
    for name in a:
        if a[name]['ground_truth']!=b[name]['ground_truth']:raise ValueError('A/B supplied annotations differ.')
    return [supplement(a[name],b[name]) for name in sorted(a)]

