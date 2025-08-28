"""Command line interface for the dagger extension runtime."""

import contextlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
import typing

import anyio

import dagger
from dagger import telemetry
from dagger.mod._exceptions import ModuleError, ModuleLoadError, record_exception
from dagger.mod._module import MAIN_OBJECT, Module

logger = logging.getLogger(__package__)

ENTRY_POINT_NAME: typing.Final[str] = "main_object"
ENTRY_POINT_GROUP: typing.Final[str] = typing.cast(str, __package__)
IMPORT_PKG: typing.Final[str] = os.getenv("DAGGER_DEFAULT_PYTHON_PACKAGE", "main")


def run(main_cls: type | None = None):
    """Entrypoint for a Python Dagger module."""
    sys.exit(anyio.run(main, main_cls))


async def main(main_cls: type | None = None) -> int | None:
    """Async entrypoint for a Dagger module."""
    async with contextlib.AsyncExitStack() as stack:
        telemetry.initialize()
        stack.callback(telemetry.shutdown)

        # Establishing connection early on to allow returning dag.error().
        # Note: if there's a connection error dag.error() won't be sent but
        # should be logged and the traceback shown on the function's stderr output.
        await stack.enter_async_context(await dagger.connect())

        try:
            mod = load_module(main_cls)
            return await mod.serve()
        except (ModuleError, dagger.QueryError) as e:
            await record_exception(e)
            return 2
        except Exception as e:
            logger.exception("Unhandled exception")
            await record_exception(e)
            return 1


def load_module(main_cls: type | None = None) -> Module:
    """Load the dagger.Module instance via the main object entry point."""
    if main_cls is None:
        ep = get_entry_point()
        try:
            main_cls = ep.load()
        except Exception as e:
            logger.exception(
                "Error while importing Python module '%s' with Dagger functions",
                ep.module,
            )
            raise ModuleLoadError(str(e)) from e

    msg = (
        "The main object must be a class decorated with @dagger.object_type, "
        f"found '{main_cls!r}'"
    )

    if not inspect.isclass(main_cls):
        raise ModuleLoadError(msg)

    try:
        mod = main_cls.__dagger_module__
    except AttributeError:
        raise ModuleLoadError(msg) from None

    mod = typing.cast(Module, mod)
    mod._main_name = main_cls.__name__  # noqa: SLF001
    return mod


def get_entry_point() -> importlib.metadata.EntryPoint:
    """Get the entry point for the main object."""
    sel = importlib.metadata.entry_points(
        group=ENTRY_POINT_GROUP,
        name=ENTRY_POINT_NAME,
    )
    if ep := next(iter(sel), None):
        return ep

    import_pkg = IMPORT_PKG

    # Fallback for modules that still use the "main" package name.
    if not importlib.util.find_spec(import_pkg):
        import_pkg = "main"

        if not importlib.util.find_spec(import_pkg):
            msg = (
                "Main object not found. You can configure it explicitly by adding "
                "an entry point to your pyproject.toml file. For example:\n"
                "\n"
                f'[project.entry-points."{ENTRY_POINT_GROUP}"]\n'
                f"{ENTRY_POINT_NAME} = '{IMPORT_PKG}:{MAIN_OBJECT}'\n"
            )
            raise ModuleLoadError(msg)

    return importlib.metadata.EntryPoint(
        group=ENTRY_POINT_GROUP,
        name=ENTRY_POINT_NAME,
        value=f"{import_pkg}:{MAIN_OBJECT}",
    )
