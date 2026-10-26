"""Exact retrospective reuse or fresh extraction; never change original banks."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_counts(old,new):
    old,new = np.asarray(old),np.asarray(new)
    if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Invalid old/new per-event counts')
    return np.array_equal(old,new)


def verify_prefix(directory,expected,baseline,candidate,dr,legacy,config):
    """Equal event vectors imply equal old/new and cross-design raw moments."""
    started = time.monotonic()
    meta_path = directory/'CHECKPOINT.json'
    if digest(meta_path) != expected['metadata_sha256']:
        raise ValueError('Prior checkpoint metadata changed')
    meta = json.loads(meta_path.read_text())
    if digest(directory/'checkpoint.npz') != meta['checkpoint_sha256'] or meta['checkpoint_sha256'] != expected['checkpoint_sha256']:
        raise ValueError('Prior checkpoint checksum changed')
    chunks = meta['completed_chunks']
    if not chunks or chunks != meta['batch']['chunks'][:len(chunks)] or chunks != expected['chunks']:
        raise ValueError('Prior checkpoint is not the recorded complete prefix')
    if len(meta['reports'])!=len(chunks) or [r['chunk_index'] for r in meta['reports']]!=chunks:
        raise ValueError('Prior reports do not match its committed prefix')
    summed_total = summed_outer = None
    proof = dict(legacy_metadata_sha256=expected['metadata_sha256'],
        legacy_checkpoint_sha256=expected['checkpoint_sha256'],chunks=chunks,
        exact_per_event_counts=True,source_files_checked=0,events_compared=0,
        new_reconstruction_sha256=candidate.reconstruction_hashes(config),reports=[],
        baseline_rg_source_sha256=digest(baseline.__file__),
        baseline_native_sha256=digest(baseline._atlas_rg.__file__),
        validator_sha256=digest(__file__))
    pp = meta['batch']['tag']=='pp'
    for report in meta['reports']:
        if report['chunk_index'] not in chunks or report['design_ids'] != ([-1] if pp else list(range(40))):
            raise ValueError('Wrong prior design/chunk identity')
        ids=[None] if pp else list(range(40))
        if [r['design_id'] for r in report['records']]!=ids:
            raise ValueError('Prior records omit, duplicate or reorder designs')
        matrices = []
        paired_identity = None
        for record in report['records']:
            for source,checksum in record['sources'].items():
                if digest(source) != checksum:
                    raise ValueError('Original source changed: '+source)
                proof['source_files_checked'] += 1
            raw = [Path(p) for p in record['sources'] if 'final_state_hadrons' in Path(p).name]
            if len(raw)!=1:
                raise ValueError('Ambiguous raw source')
            identity = legacy.fingerprints(raw[0])
            if paired_identity is None:
                paired_identity = identity
            else:
                np.testing.assert_array_equal(identity,paired_identity)
            values = []
            qa = Counter()
            for index,event in enumerate(dr.read_events(raw[0])):
                if index>=len(identity):
                    raise ValueError('Extra source event')
                np.testing.assert_array_equal([event[k] for k in legacy.FIELDS],identity[index])
                old,_ = baseline.count_event(event,legacy.BASELINE_CONFIG,'pp' if pp else 'PbPb')
                new,current_qa = candidate.count_event(event,config,'pp' if pp else 'PbPb')
                if not compare_counts(old,new):
                    raise ValueError('Cannot reuse differing counts: '+str((raw[0],index)))
                values.append(float(event['event_weight'])*np.r_[new,1.])
                qa.update(current_qa)
                proof['events_compared'] += 1
            matrix = np.asarray(values)
            if len(matrix)!=len(identity) or len(matrix)!=report['events_per_design'] or matrix.shape[1]!=40:
                raise ValueError('Prior event coverage changed')
            matrices.append(matrix)
            proof['reports'].append(dict(chunk_index=report['chunk_index'],design_id=record['design_id'],
                events=len(matrix),raw_sha256=record['sources'][str(raw[0])],cutflow=dict(qa)))
        vector = np.concatenate(matrices,axis=1)
        total,outer = vector.sum(0),vector.T@vector
        if summed_total is None:
            summed_total,summed_outer = total,outer
        else:
            summed_total += total
            summed_outer += outer
        print('RECOVERY_PREFIX_CHUNK_VERIFIED',directory.name,report['chunk_index'],flush=True)
    with np.load(directory/'checkpoint.npz',allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    np.testing.assert_allclose(summed_total,arrays['new_total'],rtol=1e-12,atol=1e-18)
    np.testing.assert_allclose(summed_outer,arrays['new_new'],rtol=1e-12,atol=1e-18)
    old_width = report['old_vector_width']
    np.testing.assert_allclose(arrays['old_total'][old_width-1::old_width],summed_total[39::40],rtol=1e-12,atol=1e-18)
    proof.update(status='ALL_EVENT_EXACT_COUNTS_AND_RAW_RG_MOMENTS_VERIFIED',
                 seconds=time.monotonic()-started)
    return arrays,deepcopy(meta['reports']),proof


def publish(output,arrays,reports,batch,root,payload,started,reuse=None):
    plan = json.loads((payload/'MANIFEST.json').read_text())
    prior_root = Path(plan['recovery']['legacy_root'])
    n=1 if batch['tag']=='pp' else 40
    shapes=dict(old_total=(n*136,),new_total=(n*40,),old_new=(n*136,n*40),new_new=(n*40,n*40))
    if n==1:shapes['old_old']=(136,136)
    if set(arrays)!=set(shapes) or any(v.shape!=shapes[k] or not np.isfinite(v).all() for k,v in arrays.items()):
        raise ValueError('Invalid/nonfinite recovered moment arrays')
    used = sum(p.stat().st_size for p in prior_root.rglob('*') if p.is_file())
    if used + sum(x.nbytes for x in arrays.values()) + 2**20 > plan['storage_cap_bytes']:
        raise ValueError('Cumulative study storage cap would be exceeded')
    pending = output/'checkpoint.pending.npz'
    np.savez_compressed(pending,**arrays)
    pending.replace(output/'checkpoint.npz')
    complete = [r['chunk_index'] for r in reports]
    if complete != batch['chunks'][:len(complete)]:
        raise ValueError('Recovery does not cover a unique contiguous batch prefix')
    from jetscape_analysis.analysis import atlas_rg as rg
    meta = dict(status='COMPLETE' if complete==batch['chunks'] else 'PARTIAL',batch=batch,
        completed_chunks=complete,reports=reports,seconds=time.monotonic()-started,
        maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        collector_sha256=digest(__file__),manifest_sha256=digest(payload/'MANIFEST.json'),
        checkpoint_sha256=digest(output/'checkpoint.npz'),new_events_generated=False,
        rg_edges=plan['recovery']['rg_config']['rg_edges'],pi0_treatment='keep',
        reconstruction_sha256=rg.reconstruction_hashes(plan['recovery']['rg_config']),
        reuse_proof=reuse,finite_moment_shapes_verified=True,
        slurm_job_id=__import__('os').environ['SLURM_JOB_ID'])
    temp = output/'CHECKPOINT.pending.json'
    temp.write_text(json.dumps(meta,indent=2)+'\n')
    temp.replace(output/'CHECKPOINT.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    args = parser.parse_args()
    payload = Path(__file__).resolve().parent
    root = payload.parent
    plan = json.loads((payload/'MANIFEST.json').read_text())
    candidates = plan['batches']+[plan['recovery']['gate_batch']]
    batch = next(b for b in candidates if b['id']==args.batch)
    output = root/'batches'/args.batch
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    from jetscape_analysis.analysis import atlas_dr12 as dr,atlas_rg as rg
    legacy = load_module('legacy_collector',payload/'legacy_collect_reduced.py')
    legacy.BASELINE_CONFIG = deepcopy(legacy.RG_CONFIG)
    legacy.RG_CONFIG = plan['recovery']['rg_config']
    baseline = load_module('legacy_rg',payload/'legacy_rg.py')
    if args.batch=='pp_00':
        previous=plan['recovery']['prefixes']['pp_reused']
        pp_arrays,pp_reports,pp_proof=verify_prefix(Path(previous['directory']),previous,
            baseline,rg,dr,legacy,legacy.RG_CONFIG)
        pp_output=root/'batches/pp_reused'
        pp_output.mkdir(exist_ok=False)
        pp_batch=dict(id='pp_reused',tag='pp',chunks=[0],gate=True)
        publish(pp_output,pp_arrays,pp_reports,pp_batch,root,payload,started,pp_proof)
    arrays,reports,reuse = None,[],None
    prior = plan['recovery']['prefixes'].get(args.batch)
    if prior is not None:
        arrays,reports,reuse = verify_prefix(Path(prior['directory']),prior,baseline,rg,dr,
                                          legacy,legacy.RG_CONFIG)
        publish(output,arrays,reports,batch,root,payload,started,reuse)
    # The failure gate's first chunk is an auxiliary result, included exactly once.
    if args.batch=='aa05_00':
        gate = root/'batches/failure_gate'
        meta = json.loads((gate/'CHECKPOINT.json').read_text())
        if meta['status']!='COMPLETE' or meta['completed_chunks']!=[12] or meta['reconstruction_sha256']!=rg.reconstruction_hashes(legacy.RG_CONFIG):
            raise ValueError('Fresh gate is incomplete or has different reconstruction')
        if digest(gate/'checkpoint.npz')!=meta['checkpoint_sha256'] or arrays is not None:
            raise ValueError('Ambiguous or changed failure-gate reuse')
        with np.load(gate/'checkpoint.npz',allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        reports = deepcopy(meta['reports'])
        reuse = dict(status='SAME_BACKEND_FRESH_GATE_REUSED',gate_sha256=digest(gate/'CHECKPOINT.json'))
        publish(output,arrays,reports,batch,root,payload,started,reuse)
    if reports and batch['chunks'][:len(reports)] != [r['chunk_index'] for r in reports]:
        raise ValueError('Invalid checkpoint prefix')
    producer = load_module('unchanged_producer',legacy.BASE/'simulation-covariance-20261007/simulation_covariance.py')
    for path,checksum in json.loads((payload/'external_pins.json').read_text()).items():
        if digest(path)!=checksum:
            raise ValueError('External input/source pin changed: '+path)
    config = dr.load_config(legacy.BASE/'atlas-dr12-pilot-20261007/config/atlas_dr12_5020.yaml')
    layout = json.loads((payload/'layout_hpt10.json').read_text())
    for chunk in batch['chunks'][len(reports):]:
        stats,report = legacy.extract_chunk(producer,batch['tag'],chunk,layout,config,batch['gate'])
        if arrays is None:
            arrays = stats
        else:
            if set(arrays)!=set(stats):
                raise ValueError('Incompatible recovery moment keys')
            for key in arrays:
                arrays[key] += stats[key]
        reports.append(report)
        publish(output,arrays,reports,batch,root,payload,started,reuse)
        print('RECOVERY_CHUNK_COMPLETE',args.batch,chunk,flush=True)
    print('RECOVERY_BATCH_COMPLETE',args.batch,flush=True)


if __name__=='__main__':
    main()
