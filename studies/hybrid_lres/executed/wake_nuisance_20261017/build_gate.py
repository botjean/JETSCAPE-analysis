"""Compile only this study's pinned selectors, inside an approved compute job."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig

import fastjet
import pybind11


def main():
    work = Path(__file__).resolve().parent.parent
    directory = work/'wake_sensitivity_20261017'
    native = work/'native'
    target = directory/('_wake_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    if target.exists():
        raise ValueError('Refuse to replace a compiled gate binary')
    fj = Path(fastjet.__file__).resolve().parent
    lib = next(fj/p for p in ('lib64', 'lib') if (fj/p/'libfastjet.so').exists())
    sources = [directory/'native.cc', native/'GroomerShop.cc', native/'LundGenerator.cc']
    command = ['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-fvisibility=hidden', '-Wl,-Bsymbolic']
    command += ['-I'+str(p) for p in
                (pybind11.get_include(), sysconfig.get_path('include'), fj/'include', native)]
    command += [str(p) for p in sources]
    command += ['-L'+str(lib), '-Wl,-rpath,'+str(lib), '-lfastjettools', '-lfastjet', '-o', str(target)]
    subprocess.run(command, check=True)
    report = dict(command=command, sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [*sources, *sorted(native.glob('*.hh')), native/'fjtools.cxx', target]},
                  runtime_sha256={str((lib/p).resolve()): hashlib.sha256((lib/p).read_bytes()).hexdigest()
                                   for p in ('libfastjet.so', 'libfastjettools.so')})
    (directory/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    print('WAKE_GATE_BUILD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
