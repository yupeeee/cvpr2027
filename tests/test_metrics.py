import json
from types import SimpleNamespace

import pytest
import torch

from utils.config import DEFAULTS, dry_run, parse_args
from utils.data import Run, atomic_json, base_row, make_samples, merge_rows, request_for, scientific_fingerprint
from utils.metrics import Cost, cluster_bootstrap, paired_bootstrap, regression_metrics, trial_metrics, weighted_mean
from utils.tasks import load_tasks


def test_trial_joint_is_per_trial_and_noop_unmet_fails():
    tasks = load_tasks('configs/tasks.json')
    ref = {'appearance':-10.0,'composition':3.0,'category':2.0}
    noop = trial_metrics(ref, ref, 'appearance', 1, tasks, 'a', True, True)
    assert noop['originally_unsatisfied'] and not noop['target_success'] and not noop['joint_success']
    changed = dict(ref, appearance=10.0)
    result = trial_metrics(ref, changed, 'appearance', 1, tasks, 'a', True, True)
    assert result['joint_success'] and result['protected_ok']
    wrong_assessment = trial_metrics(ref, changed, 'appearance', 1, tasks, 'a', True, False)
    assert wrong_assessment['target_success'] and not wrong_assessment['joint_success']
    malformed = trial_metrics(ref, dict(changed, appearance=None), 'appearance', 1, tasks, 'a', True, True)
    assert not malformed['target_success'] and malformed['originally_unsatisfied'] is None


def test_weighted_and_cluster_aggregation():
    assert weighted_mean([10,3],[2,1]) == pytest.approx(13/3)
    assert weighted_mean([],[]) is None
    rows = [{'root_seed':seed,'v':v} for seed,v in [(1,0),(1,0),(2,1),(2,1)]]
    result = cluster_bootstrap(rows, lambda r:r['v'], 7, 50, .95)
    assert result['mean']==.5 and result['roots']==2 and result['n']==4
    assert result['low'] <= .5 <= result['high']
    paired = paired_bootstrap(rows[::2], rows[::2], lambda r:r['root_seed'],lambda r:r['v'],7,20,.95)
    assert paired['mean']==0 and paired['low']==0 and paired['high']==0
    one = cluster_bootstrap(rows[:1],lambda r:r['v'],7,20,.95)
    assert one['low'] is None and one['roots']==1
    metric = regression_metrics([1,2],[1,1])
    assert metric['accuracy']==1 and metric['balanced_accuracy'] is None


def test_config_precedence_and_validation(tmp_path, capsys):
    p=tmp_path/'config.json'
    atomic_json(p,{'num_samples':11,'device':'cpu'})
    a=parse_args('collect',['--config',str(p),'--num-samples','12','--dry-run'])
    assert a.num_samples==12 and a.device=='cpu'
    assert dry_run(a,'collect')
    assert 'no models' in capsys.readouterr().out
    atomic_json(p,{'not_a_setting':3})
    with pytest.raises(SystemExit):
        parse_args('collect',['--config',str(p)])
    atomic_json(p,{'num_steps':3.5})
    with pytest.raises(SystemExit):
        parse_args('collect',['--config',str(p)])
    with pytest.raises(SystemExit):
        parse_args('collect',['--ddim-eta','1'])


class LocalDist:
    rank=0
    world_size=1
    is_main=True
    device=torch.device('cpu')
    def barrier(self):
        pass


def test_cache_split_resume_and_manifest(tmp_path):
    args=parse_args('collect',['--device','cpu','--num-samples','12','--logs-dir',str(tmp_path/'logs'),
                               '--ckpts-dir',str(tmp_path/'ckpts'),'--figs-dir',str(tmp_path/'figs')])
    samples,prompts=make_samples(args)
    assert len({s['sample_id'] for s in samples})==12
    assert all({s['split'] for s in samples if s['root_seed']==seed}.__len__()==1 for seed in range(args.seed,args.seed+12))
    changed=SimpleNamespace(**(vars(args)|{'device':'cuda','resume':True,'run_id':'another'}))
    assert make_samples(changed)==(samples,prompts)
    assert scientific_fingerprint(changed)==scientific_fingerprint(args)
    assert request_for(samples[0]['sample_id'],'appearance',args.request_seed) in (-1,1)
    run=Run(args,LocalDist(),'collect')
    row=base_row(samples[0],key='a')
    run.write_row('collect',row)
    run.write_tensor('trajectories/a.pt',{'fingerprint':run.fingerprint,'states':{0:torch.ones(1)}})
    assert run.load_trajectory('a')['states'][0].item()==1
    # Incomplete artifacts are retried automatically; --resume is now legacy.
    assert run.completed('collect') == set()
    other=Run(args,LocalDist(),'collect')
    assert other.completed('collect') == set()
    assert other.finish('collect',[])['count']==0
    with pytest.raises(ValueError,match='Duplicate'):
        merge_rows([row,row])
    with pytest.raises(ValueError,match='missing'):
        merge_rows([row],['a','b'])
    args.guidance_scale+=1
    with pytest.raises(ValueError,match='fingerprint'):
        Run(args,LocalDist(),'collect')


def test_cost_records_timing_and_counts():
    cost=Cost()
    with cost.measure('cpu'):
        cost.add('unet_samples',2)
    assert cost.to_dict()['unet_samples']==2 and cost.to_dict()['seconds']>=0


def test_component_identity_changes_reject_cache(tmp_path):
    from utils.data import model_identity, register_components
    model=tmp_path/'local_model'
    model.mkdir()
    (model/'config.json').write_text('{"layers":1}')
    (model/'weights.safetensors').write_bytes(b'offline-test-content-only')
    first=model_identity(str(model),'main','config.json')
    assert first['source']=='local'
    (model/'weights.safetensors').write_bytes(b'changed-test-content')
    second=model_identity(str(model),'main','config.json')
    assert first != second
    run=SimpleNamespace(path=tmp_path/'run',dist=LocalDist())
    sampler=SimpleNamespace(model_id='local-fixture',weights_identity=first)
    scorers={'a':SimpleNamespace(weights_identity={'source':'fixture','version':1})}
    digest=register_components(run,sampler,scorers)
    assert register_components(run,sampler,scorers)==digest
    sampler.weights_identity=second
    with pytest.raises(ValueError,match='identities differ'):
        register_components(run,sampler,scorers)


def test_nonfinite_and_duplicate_settings_rejected():
    for argv in (['--edit-budgets','nan'], ['--editors','noop','noop'], ['--coordinate-amplitude','inf']):
        with pytest.raises(SystemExit):
            parse_args('collect',argv)
