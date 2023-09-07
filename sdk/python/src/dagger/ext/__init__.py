from ._arguments import Argument as Argument
from ._module import Module as Module

_env = Module()
check = _env.check


def default_module() -> Module:
    return _env
