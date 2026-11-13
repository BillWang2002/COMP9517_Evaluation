"""Batch-one FP32 adapters with a common, synchronized timing boundary."""
from datetime import datetime,timezone
import gc
import hashlib
from importlib.metadata import version
import math
import os
import platform
import statistics
import time

from PIL import Image
from bird_project.data import collect_samples,parse_yolo_file
from bird_project.evaluate import annotation_fingerprint,canonical_boxes,validate_records
from bird_project.model import choose_device,make_model,synchronize
from .settings import CAP,DATA,FUSION_PARAMETERS,INPUT_SIZE,MODELS,ROOT,SPLITS,THREADS,TIMING_SCOPE,WARMUP
from .postprocess import supplement


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def model_hashes(model):
    names=('A1','B0') if model=='FUSION' else (model,)
    result={}
    for name in names:
        path=ROOT/MODELS[name]['checkpoint']
        if not path.is_file():raise FileNotFoundError(f'Missing checkpoint: {path}. See README.')
        result[name]=sha256(path)
        if result[name]!=MODELS[name]['sha256']:raise ValueError(f'Checkpoint differs from fixed {name} weights.')
    return result


def samples_for(split,limit=None,allow_test=False):
    if split=='test' and not allow_test:raise ValueError('Test requires --allow-test; never select parameters on test.')
    if split=='test' and limit is not None:raise ValueError('Final test must use all 240 images.')
    samples=collect_samples(DATA,split)
    if len(samples)!=SPLITS[split]['images']:raise ValueError(f'Incomplete {split} dataset.')
    # Expected filenames prevent replacing one image with a different same-count set.
    import csv
    with (ROOT/'data/splits.csv').open(newline='') as f:
        expected=sorted(r['file_name'] for r in csv.DictReader(f) if r['split']==split)
    if [s.image_path.name for s in samples]!=expected:raise ValueError('Official split filenames differ.')
    if limit is not None:
        if limit<1 or limit>len(samples):raise ValueError('--limit is outside the split size.')
        samples=samples[:limit]
    return samples


def timing_summary(seconds):
    if not seconds:raise ValueError('Cannot summarize zero timed calls.')
    ordered=sorted(seconds)
    return {'calls':len(seconds),'mean_ms':1000*statistics.mean(seconds),'median_ms':1000*statistics.median(seconds),
            'p95_ms':1000*ordered[max(0,math.ceil(.95*len(seconds))-1)],'scope':TIMING_SCOPE}


def environment(device):
    return {'platform':platform.platform(),'device':str(device),'dtype':'float32','batch_size':1,
            'input_size':INPUT_SIZE,'threads':THREADS,'warmup_calls':WARMUP,
            'versions':{k:version(k) for k in ('torch','torchvision','ultralytics','torchmetrics','pycocotools','numpy','Pillow')}}


class Detector:
    def __init__(self,name,device='auto'):
        import torch
        torch.set_num_threads(THREADS)
        if name not in (*MODELS,'FUSION'):raise ValueError('Unknown model.')
        self.name=name;self.device=choose_device(device);self.hashes=model_hashes(name)
        if self.device.type=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') not in ('0',''):
            raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK for comparable pure-MPS timing.')
        self.models={}
        for model in self.hashes:
            config=MODELS[model]
            if model=='A1':
                network=make_model()
                checkpoint=torch.load(ROOT/config['checkpoint'],map_location='cpu',weights_only=True)
                network.load_state_dict(checkpoint['model_state_dict'],strict=True)
                del checkpoint
                network.transform.min_size=(INPUT_SIZE,);network.transform.max_size=INPUT_SIZE
                network.roi_heads.score_thresh=config['score_floor'];network.roi_heads.nms_thresh=config['nms_iou']
                network.roi_heads.detections_per_img=CAP
                self.models[model]=network.to(self.device).float().eval()
            else:
                from ultralytics import YOLO
                self.models[model]=YOLO(str(ROOT/config['checkpoint']))

    def predict(self,image):
        import numpy as np
        import torch
        outputs={}
        with torch.inference_mode():
            for model,network in self.models.items():
                if model=='A1':
                    tensor=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).contiguous().float()/255
                    prediction=network([tensor.to(self.device)])[0]
                    outputs[model]={k:prediction[k].detach().cpu().tolist() for k in ('boxes','scores','labels')}
                else:
                    config=MODELS[model]
                    result=network.predict(source=image,imgsz=INPUT_SIZE,conf=config['score_floor'],iou=config['nms_iou'],
                                           max_det=CAP,device='cuda:0' if str(self.device)=='cuda' else str(self.device),
                                           quantize=32,verbose=False)[0]
                    if network.predictor.model.fp16:raise RuntimeError('Unexpected FP16 backend.')
                    classes=result.boxes.cls.detach().cpu().tolist()
                    if any(c!=0 for c in classes):raise ValueError('Expected single source class bird=0.')
                    outputs[model]={'boxes':result.boxes.xyxy.detach().cpu().tolist(),
                                    'scores':result.boxes.conf.detach().cpu().tolist(),'labels':[1]*len(classes)}
        return supplement(outputs['A1'],outputs['B0']) if self.name=='FUSION' else outputs[self.name]

    def warmup(self,images):
        for i in range(WARMUP):self.predict(images[i%len(images)])
        synchronize(self.device)

    def timed_predict(self,image):
        synchronize(self.device);started=time.perf_counter()
        output=self.predict(image);synchronize(self.device)
        return output,time.perf_counter()-started

    def close(self):
        import torch
        self.models.clear();gc.collect()
        if self.device.type=='mps':torch.mps.empty_cache()
        elif self.device.type=='cuda':torch.cuda.empty_cache()


def predict_split(name,split,device='auto',limit=None,allow_test=False):
    samples=samples_for(split,limit,allow_test)
    if split=='test' and name=='FUSION':raise ValueError('Fusion was not adopted; evaluate it on val only.')
    detector=Detector(name,device);records=[]
    try:
        warm=[]
        for sample in samples[:WARMUP]:
            with Image.open(sample.image_path) as source:warm.append(source.convert('RGB'))
        detector.warmup(warm)
        for i,sample in enumerate(samples):
            with Image.open(sample.image_path) as source:image=source.convert('RGB')
            width,height=image.size
            gt=canonical_boxes(parse_yolo_file(sample.label_path,width,height))
            output,elapsed=detector.timed_predict(image)
            records.append({'image_id':i,'file_name':sample.image_path.name,'width':width,'height':height,**output,
                            'ground_truth':{'boxes':gt,'labels':[1]*len(gt)},'inference_seconds':elapsed})
            if (i+1)%25==0 or i+1==len(samples):print(f'{name} {split}: {i+1}/{len(samples)}',flush=True)
        validate_records(records)
        return {'schema_version':2,'model_id':name,'split':split,'partial':len(samples)!=SPLITS[split]['images'],
                'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_hashes':detector.hashes,
                'environment':environment(detector.device),'model_settings':{m:MODELS[m] for m in detector.hashes},
                'postprocessing':FUSION_PARAMETERS if name=='FUSION' else None,
                'annotation_fingerprint':annotation_fingerprint(records),
                'timing':timing_summary([r['inference_seconds'] for r in records]),'records':records}
    finally:detector.close()
