"""One CLI for prediction, evaluation, postprocessing, timing and a visual demo."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics

from PIL import Image,ImageDraw,ImageFont
from bird_project.evaluate import annotation_fingerprint,write_json
from .evaluation import evaluate,load_cache,output_directory,save_results,write_csv
from .postprocess import supplement_records
from .runtime import Detector,environment,model_hashes,predict_split,samples_for,sha256,timing_summary
from .settings import FUSION_PARAMETERS,MODELS,SPLITS,WORKING_SCORES


def predict_command(args):
    out=output_directory(args.output)
    cache=predict_split(args.model,args.split,args.device,args.limit,args.allow_test)
    write_json(out/f'{args.model}_{args.split}.json',cache)
    print(json.dumps(cache['timing'],indent=2))


def evaluate_command(args):
    # Validate every requested test action before any new predictions or outputs.
    if args.split=='test' and (not args.allow_test or args.limit is not None or 'FUSION' in args.models):
        raise ValueError('Test requires --allow-test, full split, and A1/B0 only.')
    if args.limit is not None and not args.allow_partial:
        raise ValueError('Partial engineering evaluation requires --allow-partial.')
    out=output_directory(args.output);results=[]
    for model in args.models:
        if args.cache_dir:
            if args.limit is not None:raise ValueError('--limit applies to new inference, not existing caches.')
            cache=load_cache(Path(args.cache_dir)/f'{model}_{args.split}.json',args.allow_test,args.allow_partial)
            if (cache['model_id'],cache['split'])!=(model,args.split):raise ValueError('Cache filename/model/split mismatch.')
        else:
            cache=predict_split(model,args.split,args.device,args.limit,args.allow_test)
            write_json(out/f'{model}_{args.split}.json',cache)
        results.append(evaluate(cache,args.allow_partial))
    save_results(out,results)


def postprocess_command(args):
    # This experiment is deliberately restricted to validation, including when a
    # user accidentally supplies an old test cache.
    a=load_cache(args.a_cache);b=load_cache(args.b_cache)
    if (a['model_id'],b['model_id'],a['split'],b['split'])!=('A1','B0','val','val'):raise ValueError('Requires complete A1/B0 val caches.')
    out=output_directory(args.output)
    fused=deepcopy(b);fused.update(model_id='FUSION',postprocessing=deepcopy(FUSION_PARAMETERS),
                                  checkpoint_hashes={m:MODELS[m]['sha256'] for m in ('A1','B0')},
                                  model_settings={m:MODELS[m] for m in ('A1','B0')},
                                  records=supplement_records(a['records'],b['records']),timing=None)
    fused['created_utc']=datetime.now(timezone.utc).isoformat()
    fused['source_caches']={str(Path(p).resolve()):sha256(p) for p in (args.a_cache,args.b_cache)}
    fused['cache_kind']='postprocessing existing validation predictions; no new detector inference'
    # Input model times cannot stand in for an end-to-end fusion measurement.
    for record in fused['records']:record.pop('inference_seconds',None)
    fused['annotation_fingerprint']=annotation_fingerprint(fused['records'])
    fused['note']='Cache-only selective supplementation; use benchmark --models FUSION for measured live two-model latency.'
    write_json(out/'FUSION_val.json',fused)
    save_results(out,[evaluate(b),evaluate(fused)])


def benchmark_command(args):
    if args.rounds<2:raise ValueError('Repeated timing requires at least two rounds.')
    if args.rounds%2:raise ValueError('Use an even number of rounds to balance forward/reverse model order.')
    samples=samples_for('val',args.limit);images=[]
    for sample in samples:
        with Image.open(sample.image_path) as source:images.append(source.convert('RGB'))
    out=output_directory(args.output);run_rows=[];image_rows=[];environments={}
    for roundno in range(1,args.rounds+1):
        order=args.models if roundno%2 else list(reversed(args.models))
        for position,model in enumerate(order,1):
            detector=Detector(model,args.device);durations=[]
            try:
                detector.warmup(images)
                for sample,image in zip(samples,images):
                    _,seconds=detector.timed_predict(image);durations.append(seconds)
                    image_rows.append({'round':roundno,'position':position,'model':model,'file_name':sample.image_path.name,'ms':1000*seconds})
                stats=timing_summary(durations);environments[model]=environment(detector.device)
                run_rows.append({'round':roundno,'position':position,'model':model,'images':len(images),**{k:stats[k] for k in ('mean_ms','median_ms','p95_ms')}})
                print(f"round {roundno}: {model} mean={stats['mean_ms']:.3f} ms/image",flush=True)
            finally:detector.close()
    summaries=[]
    for model in args.models:
        means=[r['mean_ms'] for r in run_rows if r['model']==model]
        summaries.append({'model':model,'rounds':args.rounds,'images_per_round':len(images),
                          'partial_val':len(images)!=SPLITS['val']['images'],'mean_of_run_means_ms':statistics.mean(means),
                          'min_run_mean_ms':min(means),'max_run_mean_ms':max(means),'stdev_run_means_ms':statistics.stdev(means)})
    write_csv(out/'per_run.csv',run_rows);write_csv(out/'per_image.csv',image_rows);write_csv(out/'summary.csv',summaries)
    write_json(out/'environment.json',{'models':environments,'checkpoint_hashes':model_hashes('FUSION') if 'FUSION' in args.models else {m:model_hashes(m)[m] for m in args.models},
                                      'fusion_parameters':FUSION_PARAMETERS,'order':'listed models on odd rounds, reverse order on even rounds',
                                      'limitation':'No idle-system, thermal or power control; distributions are not training-seed confidence intervals.'})
    print(json.dumps(summaries,indent=2))


def demo_command(args):
    path=Path(args.image).expanduser();output=Path(args.output).expanduser()
    if output.exists():raise FileExistsError(f'Preserve existing demo output: {output}')
    if not path.is_file():raise FileNotFoundError(path)
    if args.score is not None and not 0<=args.score<=1:raise ValueError('--score must be in [0,1].')
    with Image.open(path) as source:image=source.convert('RGB')
    w,h=image.size;panel=Image.new('RGB',(w*(len(args.models)+1),h+62),'white')
    panel.paste(image,(0,62));draw=ImageDraw.Draw(panel);font=ImageFont.load_default(size=15)
    draw.text((10,10),'Input image',fill='#222',font=font)
    draw.text((10,32),path.name,fill='#222',font=font)
    predictions=[]
    for column,model in enumerate(args.models,1):
        detector=Detector(model,args.device)
        try:
            detector.warmup([image]);prediction,elapsed=detector.timed_predict(image)
        finally:detector.close()
        threshold=WORKING_SCORES[model] if args.score is None else args.score
        selected=[i for i,s in enumerate(prediction['scores']) if s>=threshold]
        view=image.copy();paint=ImageDraw.Draw(view)
        for i in selected:
            box=prediction['boxes'][i];color='#d37614'
            if model=='FUSION' and prediction['prediction_sources'][i]['model']=='A1':color='#2672af'
            paint.rectangle(box,outline=color,width=2)
            paint.text((box[0],max(0,box[1]-15)),f"{prediction['scores'][i]:.4f}",fill=color,font=font)
        panel.paste(view,(column*w,62))
        draw.text((column*w+8,8),f'{model} score>={threshold:.2f}: {len(selected)} detections',fill='#222',font=font)
        predictions.append({'model':model,'score':threshold,'visible_detections':len(selected),'single_call_ms':elapsed*1000})
    output.parent.mkdir(parents=True,exist_ok=True);panel.save(output)
    print(json.dumps({'input':str(path.resolve()),'output':str(output.resolve()),'predictions':predictions,
                      'note':'Illustrative prediction counts, not GT-verified bird counts; single-call times are not benchmark means.'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check',help='Check official val/test filename pairing and fixed weight hashes; no inference.')
    check.set_defaults(handler=lambda args:print(json.dumps({'splits':{s:len(samples_for(s,allow_test=True)) for s in SPLITS},'weights':model_hashes('FUSION')},indent=2)))
    def runtime_options(p):p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    def split_options(p):
        p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--limit',type=int)
        p.add_argument('--allow-test',action='store_true');runtime_options(p)
    p=commands.add_parser('predict',help='Generate full-score predictions only; no metrics.')
    p.add_argument('--model',choices=list(WORKING_SCORES),required=True);split_options(p)
    p.add_argument('--output',required=True);p.set_defaults(handler=predict_command)
    p=commands.add_parser('evaluate',help='Unified A/B/FUSION COCO and fixed working-point evaluation.')
    p.add_argument('--models',nargs='+',choices=list(WORKING_SCORES),default=['A1','B0']);split_options(p)
    p.add_argument('--cache-dir',help='Directory containing MODEL_SPLIT.json from predict/evaluate.')
    p.add_argument('--allow-partial',action='store_true',help='Engineering checks only; never formal results.')
    p.add_argument('--output',required=True);p.set_defaults(handler=evaluate_command)
    p=commands.add_parser('postprocess',help='Evaluate the fixed selective supplementation against B0; complete val caches only.')
    p.add_argument('--a-cache',required=True);p.add_argument('--b-cache',required=True);p.add_argument('--output',required=True);p.set_defaults(handler=postprocess_command)
    p=commands.add_parser('benchmark',help='Live synchronized timing; odd/even rounds reverse model order; validation only.')
    p.add_argument('--models',nargs='+',choices=list(WORKING_SCORES),default=['A1','B0','FUSION'])
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--limit',type=int);runtime_options(p)
    p.add_argument('--output',required=True);p.set_defaults(handler=benchmark_command)
    p=commands.add_parser('demo',help='Predict an arbitrary image and render side-by-side panels; no labels needed.')
    p.add_argument('--image',required=True);p.add_argument('--models',nargs='+',choices=list(WORKING_SCORES),default=['A1','B0','FUSION'])
    p.add_argument('--score',type=float,help='Display-only override; omit to use fixed per-model scores.')
    p.add_argument('--output',required=True);runtime_options(p);p.set_defaults(handler=demo_command)
    args=parser.parse_args()
    if hasattr(args,'models') and len(args.models)!=len(set(args.models)):parser.error('Models must be unique.')
    try:args.handler(args)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:parser.exit(2,f'Error: {exc}\n')
