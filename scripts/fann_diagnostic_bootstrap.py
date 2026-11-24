"""Load stdlib-only diagnostics without executing tfai's eager package imports."""
import importlib.util
from pathlib import Path


def load_diagnostic_helpers():
    root=Path(__file__).resolve().parents[1]/'tfai'/'benchmarks'
    modules=[]
    for name in ('fann_comm_diagnostics','fann_logging'):
        spec=importlib.util.spec_from_file_location('_standalone_'+name,root/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules.append(module)
    return modules[0],modules[1].print
