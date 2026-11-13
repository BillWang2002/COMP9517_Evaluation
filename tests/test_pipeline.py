"""Small numerical and CLI-contract tests; no detectors, no test-set decoding."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from bird_project.evaluate import annotation_fingerprint,calculate_metrics,canonical_boxes,diagnostic_metrics
from bird_eval.evaluation import load_cache,output_directory,validate_cache
from bird_eval.postprocess import supplement,supplement_records
from bird_eval.settings import MODELS


def record(boxes=None,scores=None,gt=None):
    boxes=boxes or []
    return {'image_id':0,'file_name':'beach_bird_1370.jpg','width':512,'height':512,
            'boxes':boxes,'scores':scores if scores is not None else [.9]*len(boxes),'labels':[1]*len(boxes),
            'ground_truth':{'boxes':gt or [],'labels':[1]*len(gt or [])}}


def partial_cache():
    r=record([[0.,0.,100.,100.]],[.9],[[0.,0.,100.,100.]])
    return {'model_id':'B0','split':'val','partial':True,'records':[r],'checkpoint_hashes':{'B0':MODELS['B0']['sha256']},
            'model_settings':{'B0':MODELS['B0']},'environment':{'input_size':640,'dtype':'float32','batch_size':1},
            'postprocessing':None,'annotation_fingerprint':annotation_fingerprint([r])}


class EvaluationTests(unittest.TestCase):
    def test_float32_normalization_preserves_real_coordinate_changes(self):
        a=record(gt=[[45.914112,20.,100.,80.]])
        b=deepcopy(a);b['ground_truth']['boxes']=canonical_boxes(b['ground_truth']['boxes'])
        self.assertEqual(annotation_fingerprint([a]),annotation_fingerprint([b]))
        b['ground_truth']['boxes'][0][0]+=.1
        self.assertNotEqual(annotation_fingerprint([a]),annotation_fingerprint([b]))

    def test_duplicate_is_fp_and_gt_matched_once_before_size_grouping(self):
        r=record([[0.,0.,100.,100.]]*2,[.9,.8],[[0.,0.,100.,100.]])
        d=diagnostic_metrics([r])
        self.assertEqual((d['tp'],d['fp'],d['fn']),(1,1,0))
        self.assertEqual(sum(v['matched_count'] for v in d['size_recall'].values()),1)

    def test_empty_labels_and_predictions_have_undefined_recall(self):
        d=diagnostic_metrics([record()]);self.assertIsNone(d['recall']);self.assertIsNone(d['precision'])
        r=record(gt=[[0.,0.,10.,10.]])
        self.assertEqual(diagnostic_metrics([r])['fn'],1)

    def test_threshold_changes_diagnostics_not_ap(self):
        r=record([[0.,0.,10.,10.]],[.4],[[0.,0.,10.,10.]])
        a=calculate_metrics([r],score_threshold=.3);b=calculate_metrics([r],score_threshold=.5)
        self.assertEqual(a['map'],b['map']);self.assertEqual(a['diagnostics']['tp'],1);self.assertEqual(b['diagnostics']['tp'],0)

    def test_custom_cap_map_is_defined_and_recall_changes(self):
        boxes=[[i*12.,0.,i*12.+8.,8.] for i in range(25)]
        m=calculate_metrics([record(boxes,[1-i*.01 for i in range(25)],boxes)],primary_max_detections=20,max_detections=40)
        self.assertGreater(m['metrics_by_max_detections']['40']['map'],.99)
        self.assertLess(m['map'],.9)


class FusionTests(unittest.TestCase):
    def test_overlap_and_prediction_area_gates_and_score_scaling(self):
        a=record([[0.,0.,100.,100.],[200.,0.,300.,100.],[350.,0.,400.,50.]],[.99,.95,.99])
        b=record([[0.,0.,100.,100.]],[.6])
        result=supplement(a,b)
        self.assertEqual(result['boxes'],[[200.,0.,300.,100.],[0.,0.,100.,100.]])
        self.assertEqual(result['scores'],[.95*.75,.6])

    def test_gt_blind_and_nonmutating(self):
        a=record([[200.,0.,300.,100.]],[.95]);b=record([[0.,0.,100.,100.]],[.6])
        before=deepcopy((a,b));first=supplement(a,b)
        self.assertEqual((a,b),before)
        a['ground_truth']=b['ground_truth']={'boxes':[[0.,0.,20.,20.]],'labels':[1]}
        second=supplement(a,b)
        for key in ('boxes','scores','labels','prediction_sources'):self.assertEqual(first[key],second[key])

    def test_empty_outputs_and_a_duplicate_suppression(self):
        self.assertEqual(supplement(record(),record())['boxes'],[])
        a=record([[0.,0.,100.,100.]]*2,[.99,.95]);b=record()
        self.assertEqual(len(supplement(a,b)['boxes']),1)

    def test_pairing_checks_gt_and_dimensions(self):
        a=record();b=record();b['width']=200
        with self.assertRaises(ValueError):supplement(a,b)
        b=record(gt=[[0.,0.,10.,10.]])
        with self.assertRaises(ValueError):supplement_records([a],[b])

    def test_cap_is_joint_and_b_wins_score_ties(self):
        b=record([[0.,0.,10.,10.]]*300,[.7]*300)
        a=record([[200.,0.,300.,100.]],[.95]);r=supplement(a,b)
        self.assertEqual(len(r['boxes']),300);self.assertEqual(r['prediction_sources'][0]['model'],'A1')
        b['scores']=[.95*.75]*300;r=supplement(a,b)
        self.assertTrue(all(x['model']=='B0' for x in r['prediction_sources']))


class CacheTests(unittest.TestCase):
    def test_partial_cache_requires_explicit_flag(self):
        cache=partial_cache()
        with self.assertRaises(ValueError):validate_cache(cache)
        self.assertIs(validate_cache(cache,True),cache)

    def test_wrong_provenance_and_protocol_rejected(self):
        for field,value in [('checkpoint_hashes',{}),('model_settings',{}),('environment',{}),('postprocessing',{}),('partial',False),('annotation_fingerprint','bad')]:
            cache=partial_cache();cache[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_cache(cache,True)

    def test_test_cache_needs_allow_and_fusion_test_is_rejected(self):
        cache=partial_cache();cache['split']='test'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cache.json';p.write_text(json.dumps(cache))
            with self.assertRaisesRegex(ValueError,'allow-test'):load_cache(p)
        cache['model_id']='FUSION'
        with self.assertRaisesRegex(ValueError,'validation-only'):validate_cache(cache)

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'out';output_directory(p);(p/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):output_directory(p)
            self.assertEqual((p/'keep.txt').read_text(),'keep')


if __name__=='__main__':unittest.main()
