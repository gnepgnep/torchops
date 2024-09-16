import torch
import os
from ._C import add
from .ops import mymul, argmax

lib_dir = os.path.dirname(os.path.realpath(__file__))
so_name = None

for name in os.listdir(lib_dir):
    if name.startswith('_C') and name.endswith('.so'):
        so_name = name
        break

if not so_name:
    raise Exception('cannot find torch extension lib, '
                    'whose name starts with "_C" and ends with ".so"')

lib_path = os.path.join(os.path.dirname(__file__), so_name)

_HAS_OPS = False

try:
    torch.ops.load_library(lib_path)
    _HAS_OPS = True
except (ImportError, OSError) as e:
    print(repr(e))


__all__ = [
    "mymul",
    "argmax",
    "add"
]