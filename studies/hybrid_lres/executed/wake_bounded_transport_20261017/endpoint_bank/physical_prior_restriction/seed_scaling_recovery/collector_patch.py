"""Install the narrow seed repair on one collector's actual geometry/context."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_collector_same_ray_seed_scaling', HERE/'seed_scaling.py')
seed_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_scaling)
_MARKER = '_seed_scaling_collector_patch'
_REPORT_FIELDS = ('numerical_seed_policy', 'numerical_seed_repairs')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(col):
    """Wrap only the live collector entrypoints and its resolved rapidity seed."""
    if hasattr(col, _MARKER):
        raise ValueError('Collector seed rescaling is already installed')
    try:
        project = col.deform.__globals__['project']
        geometry = project.__globals__['geometry']
        original_seed = geometry.rapidity_seed
        original_install, original_collect = col.install_reconstruction, col.collect_one
    except (AttributeError, KeyError) as error:
        raise ValueError('Cannot resolve collector deform/project/geometry routing') from error
    if not callable(original_install) or not callable(original_collect):
        raise ValueError('Collector entrypoints are not callable')
    if not hasattr(original_seed, '__code__'):
        raise ValueError('Geometry rapidity seed is already wrapped or not a source function')
    sources = {
        'collector_patch.py': sha(__file__),
        'seed_scaling.py': sha(seed_scaling.__file__),
        'original_geometry': sha(original_seed.__code__.co_filename),
    }
    state = dict(wrapper=None, collection=None)

    def event_identity(event):
        collection = state['collection']
        if collection is None or not isinstance(event, dict):
            raise RuntimeError('Seed repair has no active source-event identity')
        if (any(event.get(key) != collection[key] for key in ('tag', 'chunk', 'design'))
                or type(event.get('event_ID')) is not int
                or isinstance(event.get('a'), (bool, np.bool_))
                or not isinstance(event.get('a'), (int, float, np.integer, np.floating))
                or not np.isfinite(event['a']) or event['a'] not in collection['nodes']):
            raise RuntimeError('Seed repair has an unknown source-event identity')
        return event

    def context():
        wrapper = state['wrapper']
        if wrapper is None:
            raise RuntimeError('Seed repair has no live reconstruction wrapper')
        return event_identity(wrapper.context)

    repair = seed_scaling.install(geometry, context=context)

    def install_reconstruction(*args, **kwargs):
        if state['collection'] is not None:
            raise RuntimeError('Cannot replace reconstruction during a collection')
        result = original_install(*args, **kwargs)
        wrapper, _ = result
        if not hasattr(wrapper, 'context'):
            raise RuntimeError('Reconstruction wrapper has no live event context')
        state['wrapper'] = wrapper
        return result

    def collect_one(*args, **kwargs):
        if state['collection'] is not None:
            raise RuntimeError('Collector seed repair does not permit nested collection')
        repair.repairs.clear()
        wrapper = kwargs.get('wrapper')
        expected = kwargs.get('context')
        config = kwargs.get('config') if 'config' in kwargs else args[2] if len(args) > 2 else None
        if (wrapper is None or wrapper is not state['wrapper'] or not isinstance(expected, dict)
                or expected.get('tag') not in ('aa05', 'aa510')
                or type(expected.get('chunk')) is not int or type(expected.get('design')) is not int
                or not isinstance(config, dict) or expected['design'] not in config.get('design_ids', [])
                or not isinstance(config.get('a_nodes'), (list, tuple)) or not config['a_nodes']):
            raise RuntimeError('Collection lacks a canonical live wrapper/source context')
        if (col.deform.__globals__.get('project') is not project
                or project.__globals__.get('geometry') is not geometry or geometry.rapidity_seed is not repair):
            raise RuntimeError('Installed seed-repair routing changed')
        state['collection'] = dict(tag=expected['tag'], chunk=expected['chunk'],
                                  design=expected['design'], nodes=tuple(config['a_nodes']))
        try:
            counts, fingerprints, report = original_collect(*args, **kwargs)
            records = deepcopy(repair.repairs)
            if records:
                identities = np.asarray(fingerprints)
                if (identities.ndim != 2 or identities.shape[1] < 1
                        or not np.isfinite(identities[:, 0]).all()
                        or np.any(identities[:, 0] != np.rint(identities[:, 0]))):
                    raise RuntimeError('Seed repairs have no valid returned event fingerprints')
                known = {int(value) for value in identities[:, 0]}
                for row in records:
                    event = event_identity(row.get('event'))
                    if row.get('policy') != seed_scaling.POLICY or event['event_ID'] not in known:
                        raise RuntimeError('Seed repair references an unknown returned event')
            if any(key in report for key in _REPORT_FIELDS):
                raise RuntimeError('Collector report already contains seed-repair provenance')
            result = dict(report, numerical_seed_policy=dict(policy=seed_scaling.POLICY,
                          source_sha256=dict(sources)), numerical_seed_repairs=records)
            return counts, fingerprints, result
        finally:
            state['collection'] = None

    col.install_reconstruction, col.collect_one = install_reconstruction, collect_one
    setattr(col, _MARKER, repair)
    return repair
