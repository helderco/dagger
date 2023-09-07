# ruff: noqa: BLE001
import json
import logging
import sys

import anyio
import cattrs
from rich.console import Console

import dagger
from dagger.log import configure_logging

from ._checks import CheckResolver
from ._converter import make_converter
from ._exceptions import FatalError, InternalError, UserError
from ._resolver import Resolver
from ._utils import asyncify, transform_error

errors = Console(stderr=True, style="bold red")
logger = logging.getLogger(__name__)


class Module:
    """Builder for a :py:class:`dagger.Module`.

    Arguments
    ---------
    log_level:
        Configure logging with this minimal level. If `None`, logging
        is not configured.
    """

    # TODO: Hook debug from `--debug` flag in CLI?
    # TODO: Default logging to logging.WARNING before release.
    def __init__(self, *, log_level: int | str | None = logging.DEBUG):
        self._log_level = log_level
        self._converter: cattrs.Converter = make_converter()
        self._resolvers: dict[str, Resolver] = {}
        self._mod = dagger.current_environment()

        # TODO: Need docstring, name and right overload to show correctly in IDE.
        self.check = CheckResolver.to_decorator(self)

    def add_resolver(self, resolver: Resolver):
        self._resolvers[resolver.graphql_name] = resolver

    def __call__(self) -> None:
        if self._log_level is not None:
            configure_logging(self._log_level)
        anyio.run(self._run)

    async def _run(self):
        async with await dagger.connect():
            await self._serve()

    async def _serve(self):
        name = await self._mod.entrypoint_input().name()
        result, exit_code = await (self._call(name) if name else self._register())
        try:
            output = json.dumps(result)
        except (TypeError, ValueError) as e:
            msg = f"Failed to serialize result: {e}"
            raise InternalError(msg) from e
        logger.debug("output => %s", repr(output))
        await self._mod.return_entrypoint_value(output)

        if exit_code:
            sys.exit(exit_code)

    async def _register(self) -> tuple[str, int]:
        # Resolvers are collected on import time, but only actually registered
        # during "serve".
        mod = self._mod

        for r in self._resolvers.values():
            try:
                mod = r.register(mod)
            except TypeError as e:
                msg = f"Failed to register function `{r.name}`: {e}"
                raise UserError(msg) from e
            logger.debug("registered => %s", r.name)

        return await mod.id(), 0

    async def _call(self, name: str) -> tuple[str, int]:
        try:
            resolver = self._resolvers[name]
        except KeyError as e:
            msg = f"Unable to find function “{name}”"
            raise FatalError(msg) from e

        logger.debug("resolver => %s", resolver.name)

        args_str = await self._mod.entrypoint_input().args()

        # Use `json` directly for more granular control over the error.
        try:
            raw_args = json.loads(args_str) or {}
        except ValueError as e:
            msg = f"Unable to decode args: {e}"
            raise InternalError(msg) from e

        logger.debug("input args => %s", repr(raw_args))

        if not isinstance(raw_args, dict):
            msg = f"Expected input args to be a JSON object, got {type(raw_args)}"
            raise InternalError(msg)

        # Serialize/deserialize from here as this is the boundary that
        # manages the lifecycle through the API.
        kwargs = await resolver.convert_arguments(self._converter, raw_args)
        logger.debug("structured args => %s", repr(kwargs))

        try:
            result = await resolver(**kwargs)
        except Exception as e:
            logger.exception("Error during function execution")
            return str(e), 1

        logger.debug("result => %s", repr(result))

        try:
            result = await asyncify(
                self._converter.unstructure,
                result,
                resolver.return_type,
            )
        except Exception as e:
            msg = transform_error(
                e,
                "Failed to unstructure result",
                resolver.wrapped_func,
            )
            raise UserError(msg) from e

        logger.debug("unstructured result => %s", repr(result))

        return result, 0
