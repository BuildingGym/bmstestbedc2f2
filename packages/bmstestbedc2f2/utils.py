import os as _os_
import sys as _sys_
import inspect as _inspect_
from types import ModuleType
from pathlib import Path


def resolve_path(
    x: str,
    base: str | Path | ModuleType | None = None
) -> str:
    """
    Resolve a relative path to a base path or a module.

    :param x: Path to resolve.
    :param base: 
        Base directory, Path object, or module. 
        Defaults to the caller file's directory.
    :return: Resolved absolute path.
    """

    if _os_.path.isabs(x):
        return x

    if base is None:
        caller_frame = _inspect_.stack()[1]
        caller_module = _sys_.modules.get(caller_frame.frame.f_globals['__name__'])
        if caller_module and hasattr(caller_module, '__file__'):
            base = caller_module
        else:
            base = _os_.getcwd()
    
    if isinstance(base, ModuleType):
        base = _os_.path.dirname(base.__file__)
    elif isinstance(base, Path):
        base = str(base)

    return _os_.path.join(base, x)
