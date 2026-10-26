"""Add low-pT event moments with unchanged reconstruction and exact old-block checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def old_projection(old_info, new_report):
    lookup={tuple(feature):i for i,feature in enumerate(new_report['features'])}
    indices=[lookup[tuple(feature)] for feature in old_info['features']]
    n=len(new_report['features'])
    indices.extend(n+new_report['normalization_features'].index(name)
                   for name in old_info['normalization_features'])
    indices.append(n+len(new_report['normalization_features']))
    return np.asarray(indices,dtype=int)


def extract(producer, paths, layout, info, config, previous):
    base=producer.EventMoments
    captured=[]
    class Capture(base):
        def update(self, counts, weight):
            super().update(counts,weight)
            captured.append(weight*np.r_[counts,1.])
    producer.EventMoments=Capture
    try:
        arrays,report=producer.extract(*paths,layout,info['system'],info['centrality'],config)
    finally:
        producer.EventMoments=base
    if report['partial']:
        raise ValueError('Cannot accept partial extraction')
    for field in ('dr12_version','dr12_source_sha256'):
        if report[field]!=info[field]:
            raise ValueError('Native reconstruction changed: '+field)
    projection=old_projection(info,report)
    vectors=np.asarray(captured)
    np.testing.assert_allclose(vectors.sum(0),arrays['total'],rtol=1e-10,atol=1e-18)
    np.testing.assert_allclose(vectors.T@vectors,arrays['outer'],rtol=1e-10,atol=1e-18)
    for name in ('total','outer','jet_sumw2'):
        use=projection if name!='jet_sumw2' else projection[:len(info['features'])]
        recovered=arrays[name][np.ix_(use,use)] if name=='outer' else arrays[name][use]
        np.testing.assert_allclose(recovered,previous[name],rtol=1e-10,atol=1e-18,
                                   err_msg='Original moments changed: '+name)
    return arrays,report,vectors,projection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','producer','layout','old-joint','dr12-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--centrality',choices=['aa05','aa510'],required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite a gate output')
    spec=importlib.util.spec_from_file_location('unchanged_low_pt_producer',args.producer)
    producer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    from jetscape_analysis.analysis import atlas_dr12 as dr
    import pyarrow.parquet as pq
    layout=json.loads(args.layout.read_text())
    config=dr.load_config(args.dr12_config)
    interval=[0,5] if args.centrality=='aa05' else [5,10]
    vectors,records=[],[]
    fingerprint=projection=None
    started=time.monotonic()
    for design in range(40):
        directory=args.source/f'point_{design:02d}_{args.centrality}'
        info_path=directory/'pooled.json'
        info=json.loads(info_path.read_text())
        if info['partial'] or info['design_id']!=design or info['centrality']!=interval:
            raise ValueError('Wrong original sample identity')
        if info['extraction_sha256']!=producer.digest(args.producer):
            raise ValueError('Unchanged producer source pin failed')
        record=info['records'][0]
        paths=[]
        for marker in ('_observables_','final_state_hadrons','_cross_section_'):
            matches=[Path(path) for path in record['sources'] if marker in Path(path).name]
            if len(matches)!=1:
                raise ValueError('Ambiguous source: '+marker)
            paths.append(matches[0])
        fields=['event_ID','event_weight','pt_hat','vertex_x','vertex_y']
        table=pq.read_table(paths[1],columns=fields)
        current=np.column_stack([table[field].to_numpy() for field in fields])
        if not np.isfinite(current).all():
            raise ValueError('Nonfinite event fingerprint')
        if fingerprint is None:
            fingerprint=current
        else:
            np.testing.assert_array_equal(current,fingerprint)
        old_chunk=directory/'chunk_0000.npz'
        with np.load(old_chunk,allow_pickle=False) as previous:
            arrays,report,captured,mapping=extract(producer,paths,layout,info,config,previous)
        assert len(info['features'])==102 and len(report['features'])==134
        if report['sources']!=record['sources']:
            raise ValueError('Original event files changed')
        if len(captured)!=len(fingerprint):
            raise ValueError('Event alignment changed')
        if projection is None:
            projection=mapping
        else:
            np.testing.assert_array_equal(projection,mapping)
        vectors.append(captured)
        records.append(dict(design_id=design,source_hashes=report['sources'],
            old_metadata_sha256=producer.digest(info_path),old_chunk_sha256=producer.digest(old_chunk),
            events=report['events'],features=report['features'],
            normalization_features=report['normalization_features'],
            dr12_source_sha256=report['dr12_source_sha256']))
        print('LOW_PT_DESIGN_COMPLETE',design,len(captured),flush=True)
    matrix=np.concatenate(vectors,axis=1)
    total,outer=matrix.sum(0),matrix.T@matrix
    old_indices=np.concatenate([projection+design*matrix.shape[1]//40 for design in range(40)])
    task=0 if args.centrality=='aa05' else 200
    old_joint=args.old_joint/f'chunk_{task:04d}.npz'
    with np.load(old_joint,allow_pickle=False) as previous:
        np.testing.assert_allclose(total[old_indices],previous['total'],rtol=1e-10,atol=1e-18)
        np.testing.assert_allclose(outer[np.ix_(old_indices,old_indices)],previous['outer'],rtol=1e-10,atol=1e-18)
    sd=np.sqrt(np.diag(outer))
    active=sd>0
    # Test a fixed deterministic principal block cheaply; full-bank PSD is an export gate.
    subset=np.flatnonzero(active)[::max(1,int(active.sum())//512)]
    correlation=outer[np.ix_(subset,subset)]/np.outer(sd[subset],sd[subset])
    minimum=float(np.linalg.eigvalsh(correlation).min())
    if minimum<-1e-10:
        raise ValueError('Raw Gram principal block failed PSD')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,total=total,outer=outer,design_ids=np.arange(40),
                        events=len(fingerprint),vector_width=vectors[0].shape[1])
    report=dict(status='LOW_PT_GATE_COMPLETE',centrality=interval,events=len(fingerprint),
        records=records,features=134,original_features=102,old_block_reproduced=True,
        paired_event_fingerprints_identical=True,seconds=time.monotonic()-started,
        raw_correlation_principal_min_eigenvalue=minimum,
        old_joint_sha256=producer.digest(old_joint),producer_sha256=producer.digest(args.producer),
        collector_sha256=producer.digest(Path(__file__)),layout_sha256=producer.digest(args.layout),
        output_sha256=producer.digest(args.output),new_event_generation=False)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('LOW_PT_GATE_COMPLETE',report['seconds'],flush=True)


if __name__=='__main__':
    main()
