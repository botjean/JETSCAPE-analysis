"""Local-only checks of the upstream hardest-root convention, not bank approval."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
ENDPOINT = HERE.parent
WORK = ENDPOINT.parents[1]
RUNTIME = ENDPOINT/'packaged_check/runtime'
SOURCE = HERE/'dr12_policy.cc'
ORIGINAL = WORK/'wake_paper_morph_20261017/dr12_diagnostic.generated.cc'
ORIGINAL_SHA256 = '3414970adab6bf75df305cef446326a7eef67ffce873bc071bd04ecd03b61ef9'
DIAGNOSTICS = {
    3287: ENDPOINT/'recovery_prepared_runtime/failure_event3287',
    4409: ENDPOINT/'signed_root_recovery/failure_event4409',
}
ROW_KEYS = ('rows_a0p0', 'rows_a0p25', 'rows_a0p5', 'rows_a0p75', 'rows_a1p0')
MODES = (('signed', -1), ('legacy_reference', -1), ('legacy_reference', 0))
GROOMERS = ('lund', 'signed_prongs', 'softdrop_reference')
OLD_KEYS = ('jets', 'cutflow', 'unresolved')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_result(result):
    return {key: result[key] for key in OLD_KEYS}


def multiset(rows):
    return sorted(tuple(row) for row in rows)


def test_original_unchanged_and_only_selection_audit_added():
    assert sha(ORIGINAL) == ORIGINAL_SHA256
    old = ORIGINAL.read_text()
    new = SOURCE.read_text().replace('#include <fastjet/tools/Recluster.hh>\n', '')
    new = new.replace('  py::list remnant_audit;\n', '')
    new = new.replace('  result["remnant_audit"] = remnant_audit;\n', '')
    first = new.index('    if (kt_jets.empty()) {')
    last = new.index('    bool accepted = false;', first)
    old_first = old.index('    if (kt_jets.size() != 1) {')
    old_last = old.index('    bool accepted = false;', old_first)
    block = new[first:last]
    assert 'Recluster::keep_only_hardest' in block
    assert 'selector.generate_output_jet(kt_jets, false)' in block
    assert new[:first] == old[:old_first]
    assert new[last:] == old[old_last:]


# The reference is built from the frozen original, not the new implementation.
# Its operator path rejoins INPUT tracks in their original order. Rejoining the
# already-clustered roots can permute tied histories; that is tested separately.
REFERENCE_HELPERS = r'''
py::dict describe_test_root(const PseudoJet &p) {
  py::dict out;
  out["p4"] = std::array<double, 4>{p.px(), p.py(), p.pz(), p.E()};
  out["sign"] = p.user_index() < 0 ? -1 : 1;
  out["pt"] = p.pt();
  out["m2"] = p.m2();
  std::vector<std::array<double, 5>> leaves;
  for (const auto &q : p.constituents())
    leaves.push_back({q.px(), q.py(), q.pz(), q.E(), q.user_index() < 0 ? -1. : 1.});
  out["leaves"] = leaves;
  return out;
}

py::dict select_test_roots(const std::vector<std::array<double, 5>> &rows) {
  SignedRecombiner recombiner;
  fastjet::JetDefinition definition(fastjet::kt_algorithm, 2.5);
  definition.set_recombiner(&recombiner);
  std::vector<PseudoJet> particles;
  for (const auto &r : rows) {
    PseudoJet p(r[0], r[1], r[2], r[3]);
    p.set_user_index(r[4] < 0 ? -1 : 0);
    particles.push_back(p);
  }
  fastjet::ClusterSequence sequence(particles, definition);
  auto roots = sequence.inclusive_jets();
  std::vector<PseudoJet> ordered_inputs(sequence.jets().begin(),
                                       sequence.jets().begin() + particles.size());
  py::list forest;
  for (const auto &p : roots) forest.append(describe_test_root(p));
  fastjet::Recluster upstream(definition, false, fastjet::Recluster::keep_only_hardest);
  auto same_sequence = upstream.generate_output_jet(roots, false);
  auto input_operator = upstream(fastjet::join(ordered_inputs));
  auto roots_operator = upstream(fastjet::join(roots));
  py::dict out;
  out["forest"] = forest;
  out["existing"] = describe_test_root(same_sequence);
  out["input_operator"] = describe_test_root(input_operator);
  out["roots_operator"] = describe_test_root(roots_operator);
  return out;
}
'''


def reference_source():
    source = ORIGINAL.read_text()
    source = source.replace('#include <fastjet/ClusterSequence.hh>',
                            '#include <fastjet/ClusterSequence.hh>\n'
                            '#include <fastjet/tools/Recluster.hh>')
    source = source.replace('// Each input row:', REFERENCE_HELPERS+'\n// Each input row:')
    source = source.replace('int legacy_marker = -1) {',
                            'int legacy_marker = -1, bool use_operator = true) {')
    source = source.replace('  for (const auto &large : fastjet::sorted_by_pt',
                            '  py::list selections;\n'
                            '  for (const auto &large : fastjet::sorted_by_pt')
    source = source.replace('if (kt_jets.size() != 1)', 'if (kt_jets.empty())')
    source = source.replace('    PseudoJet current = kt_jets[0], a, b;', r'''
    fastjet::Recluster upstream(kt_def, false, fastjet::Recluster::keep_only_hardest);
    std::vector<PseudoJet> ordered_inputs(kt_cs.jets().begin(),
                                         kt_cs.jets().begin() + associated.size());
    PseudoJet current = use_operator ? upstream(fastjet::join(ordered_inputs))
                                    : upstream.generate_output_jet(kt_jets, false);
    PseudoJet a, b;
    py::dict selected = describe_test_root(current);
    selected["jet_pt"] = jet.pt();
    selected["fraction"] = fraction;
    py::list forest;
    for (const auto &p : kt_jets) forest.append(describe_test_root(p));
    selected["forest"] = forest;
    selections.append(selected);
''')
    source = source.replace('  result["jets"] = output;',
                            '  result["selections"] = selections;\n  result["jets"] = output;')
    source = source.replace('PYBIND11_MODULE(_wake_dr12_diagnostic, m)',
                            'PYBIND11_MODULE(_upstream_policy_reference, m)')
    source = source.replace('  m.def("reconstruct",',
                            '  m.def("select_roots", &select_test_roots);\n  m.def("reconstruct",')
    source = source.replace('py::arg("legacy_marker") = -1);',
                            'py::arg("legacy_marker") = -1, py::arg("use_operator") = true);')
    return source


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    fastjet = pytest.importorskip('fastjet', reason='Native analysis environment required')
    pybind11 = pytest.importorskip('pybind11', reason='Test-only FastJet reference is compiled locally')
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    directory = tmp_path_factory.mktemp('upstream_hardest_reference')
    binary = directory/('_upstream_policy_reference'+suffix)
    package = Path(fastjet.__file__).parent
    library = package/'lib64'
    command = shlex.split(sysconfig.get_config_var('CXX') or 'c++') + [
        '-O2', '-std=c++17', '-shared', '-fPIC',
        '-I'+pybind11.get_include(), '-I'+sysconfig.get_path('include'),
        '-I'+str(package/'include'), '-x', 'c++', '-',
        '-L'+str(library), '-Wl,-rpath,'+str(library),
        '-lfastjetcontribfragile', '-lfastjettools', '-lfastjet', '-o', str(binary),
    ]
    subprocess.run(command, input=reference_source(), text=True, check=True,
                   capture_output=True, timeout=90)
    official_before = sys.modules.get('_wake_dr12_diagnostic')
    modules = {}
    for label, path in (
        ('policy', HERE/'local_native'/('_wake_dr12_diagnostic'+suffix)),
        ('original', RUNTIME/'wake_paper_morph_20261017'/('_wake_dr12_diagnostic'+suffix)),
    ):
        name = '_upstream_policy_test_'+label
        parent = ModuleType(name)
        parent.__path__ = []
        sys.modules[name] = parent
        modules[label] = load(name+'._wake_dr12_diagnostic', path)
    modules['reference'] = load('_upstream_policy_reference', binary)
    assert sys.modules.get('_wake_dr12_diagnostic') is official_before
    return SimpleNamespace(**modules)


@pytest.fixture(scope='module')
def diagnostic_rows():
    rows = {}
    for event, directory in DIAGNOSTICS.items():
        with np.load(directory/'diagnostic_rows.npz', allow_pickle=False) as data:
            assert set(data.files) == set(ROW_KEYS)
            rows[event] = {key: data[key].tolist() for key in ROW_KEYS}
    return rows


def check_audit(actual, reference):
    forests = [item for item in reference['selections'] if len(item['forest']) > 1]
    assert len(actual['remnant_audit']) == len(forests)
    for audit, selected in zip(actual['remnant_audit'], forests):
        assert audit['jet_pt'] == selected['jet_pt']
        assert audit['fraction'] == selected['fraction']
        assert audit['selected_root_p4'] == selected['p4']
        assert audit['selected_root_sign'] == selected['sign']
        assert audit['selected_root_pt'] == selected['pt']
        assert audit['selected_root_m2'] == selected['m2']
        expected = [root['p4']+[root['sign']] for root in selected['forest']]
        chosen = selected['p4']+[selected['sign']]
        expected.remove(chosen)
        assert multiset(audit['remnants']) == multiset(expected)
        assert audit['equal_pt_roots'] == sum(
            root['pt'] == selected['pt'] for root in selected['forest'])


def check_reference(native, rows, **kwargs):
    actual = native.policy.reconstruct(rows, **kwargs)
    existing = native.reference.reconstruct(rows, use_operator=False, **kwargs)
    operator = native.reference.reconstruct(rows, use_operator=True, **kwargs)
    assert set(actual) == {*OLD_KEYS, 'remnant_audit'}
    assert old_result(actual) == old_result(existing)
    assert old_result(actual) == old_result(operator)
    assert existing['selections'] == operator['selections']
    check_audit(actual, existing)
    old = native.original.reconstruct(rows, **kwargs)
    if not old['unresolved']:
        assert old_result(actual) == old
        assert actual['remnant_audit'] == []
    return actual


def test_only_reconstruct_exported(native):
    assert [key for key in dir(native.policy) if not key.startswith('_')] == ['reconstruct']
    assert native.policy.reconstruct.__doc__ == native.original.reconstruct.__doc__


@pytest.mark.parametrize('event', (3287, 4409))
@pytest.mark.parametrize('row_key', ROW_KEYS)
@pytest.mark.parametrize('groomer', GROOMERS)
@pytest.mark.parametrize(('mode', 'marker'), MODES)
def test_real_diagnostics_all_nodes_modes(native, diagnostic_rows, event, row_key,
                                        groomer, mode, marker):
    check_reference(native, diagnostic_rows[event][row_key], groomer=groomer,
                    recombiner_mode=mode, legacy_marker=marker)


@pytest.mark.parametrize('row_key', ROW_KEYS)
def test_event4409_ordinary_grooming_failure_not_fabricated_bin(native, diagnostic_rows, row_key):
    result = native.policy.reconstruct(diagnostic_rows[4409][row_key])
    assert not result['unresolved']
    assert result['jets'] == []
    assert result['cutflow']['charged_fraction_pass'] > 0
    assert result['cutflow']['grooming_failed'] == result['cutflow']['charged_fraction_pass']


@pytest.fixture(scope='module')
def fixture_rows():
    # Keep packaged helper aliases out of the workspace pytest interpreter.
    code = r'''
import importlib.util, json, pathlib, sys
runtime = pathlib.Path(sys.argv[1])
path = runtime/'wake_bounded_transport_20261017/bank/checkpoint_collect.py'
spec = importlib.util.spec_from_file_location('_policy_fixture_helper', path)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
col = helper.load_collector()
before = col.method_sources()
fixtures = col.runtime.read_fixtures(runtime/'wake_bounded_transport_20261017/bank/runtime_cases.npz')
assert len(fixtures) == 5
rows = []
for label, _, event in fixtures:
    for a in (0., .25, .5, .75, 1.):
        modified, qa = col.deform(event, a)
        assert qa['whole_event_fallback'] == 0
        rows.append((label, a, col.reader.particle_rows(modified, 'PbPb')))
assert col.method_sources() == before
print('FIXTURE_ROWS_JSON='+json.dumps(rows))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run = subprocess.run([sys.executable, '-c', code, str(RUNTIME)], check=True,
                         text=True, capture_output=True, env=env, timeout=90)
    lines = [line for line in run.stdout.splitlines() if line.startswith('FIXTURE_ROWS_JSON=')]
    assert len(lines) == 1
    result = json.loads(lines[0].split('=', 1)[1])
    assert len(result) == 25
    return result


@pytest.mark.parametrize('case', range(25))
@pytest.mark.parametrize('groomer', GROOMERS)
@pytest.mark.parametrize(('mode', 'marker'), MODES)
def test_25_original_fixture_nodes_all_modes(native, fixture_rows, case, groomer, mode, marker):
    _, _, rows = fixture_rows[case]
    check_reference(native, rows, groomer=groomer, recombiner_mode=mode, legacy_marker=marker)


# Selector-only edge cases deliberately include non-timelike roots. They are not
# claimed as detector-selected end-to-end events. All associated-input coverage
# above uses actual event rows, with unchanged native detector/track cuts.
@pytest.mark.parametrize(('rows', 'sign', 'spacelike'), (
    ([[80., 0., 0., 80., 1], [90., 0., 1., 1., -1]], -1, True),
    ([[80., 0., 0., 80., 1], [110., 0., 1., 1., 1]], 1, True),
))
def test_upstream_negative_or_spacelike_hardest_is_not_vetoed(native, rows, sign, spacelike):
    result = native.reference.select_roots(rows)
    assert len(result['forest']) == 2
    assert result['existing'] == result['input_operator']
    assert result['existing']['sign'] == sign
    assert (result['existing']['m2'] < 0) is spacelike


@pytest.mark.parametrize('reverse', (False, True))
def test_equal_pt_uses_upstream_input_history_not_positive_or_unique_rule(native, reverse):
    rows = [[90., 0., 0., 90., 1], [90., 0., 1., 1., -1]]
    if reverse:
        rows.reverse()
    result = native.reference.select_roots(rows)
    assert len(result['forest']) == 2
    assert [root['pt'] for root in result['forest']] == [90., 90.]
    assert result['existing'] == result['input_operator']
    assert result['existing']['sign'] in (-1, 1)
    # Rejoining roots can change the tie history. Check the operator is selecting
    # an actual maximum, not falsely asserting a fixed tied-root identity.
    assert result['roots_operator']['pt'] == 90.
    assert result['roots_operator'] in result['forest']


@pytest.mark.parametrize('kwargs', (
    {'ghost_pt': 0}, {'ghost_pt': -1e-9}, {'ghost_pt': 1e-5},
    {'ghost_pt': float('nan')}, {'ghost_pt': float('inf')},
    {'recombiner_mode': 'bad'}, {'groomer': 'bad'}, {'legacy_marker': 1},
))
def test_configuration_guards_unchanged(native, kwargs):
    for module in (native.policy, native.original):
        with pytest.raises(ValueError) as caught:
            module.reconstruct([], **kwargs)
        if module is native.policy:
            message = str(caught.value)
        else:
            assert str(caught.value) == message


@pytest.mark.parametrize(('index', 'value'), (
    (0, float('nan')), (1, float('inf')), (3, -1.), (4, 0.), (4, 2.),
    (5, float('nan')), (6, float('inf')),
))
def test_particle_guards_unchanged(native, index, value):
    row = [10., 0., 0., 10., 1., 1., 1.]
    row[index] = value
    for module in (native.policy, native.original):
        with pytest.raises(ValueError) as caught:
            module.reconstruct([row])
        if module is native.policy:
            message = str(caught.value)
        else:
            assert str(caught.value) == message
