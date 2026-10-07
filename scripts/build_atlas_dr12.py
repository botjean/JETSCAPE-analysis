"""Build the optional native module against the isolated pip FastJet installation."""

import os
from pathlib import Path
import shlex
import subprocess
import sysconfig

import fastjet
import pybind11

root = Path(__file__).resolve().parents[1]
fj = Path(fastjet.__file__).resolve().parent
lib = next((fj / name for name in ("lib64", "lib") if (fj / name / "libfastjet.so").exists()), None)
if lib is None:
    raise RuntimeError("Expected a Linux FastJet wheel containing libfastjet.so")
target = root / "jetscape_analysis/analysis" / ("_atlas_dr12" + sysconfig.get_config_var("EXT_SUFFIX"))
command = shlex.split(os.environ.get("CXX", "c++")) + [
    "-O2",
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-Wall",
    "-Wextra",
    f"-I{pybind11.get_include()}",
    f"-I{sysconfig.get_path('include')}",
    f"-I{fj / 'include'}",
    str(root / "jetscape_analysis/analysis/atlas_dr12.cc"),
    f"-L{lib}",
    f"-Wl,-rpath,{lib}",
    "-lfastjetcontribfragile",
    "-lfastjettools",
    "-lfastjet",
    "-o",
    str(target),
]
subprocess.run(command, check=True)
print(target)
