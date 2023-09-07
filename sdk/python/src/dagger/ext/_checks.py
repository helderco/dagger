from types import NoneType
from typing import TypeAlias

from typing_extensions import override

import dagger

from ._resolver import Resolver

CheckReturnType: TypeAlias = str | NoneType | dagger.Check | dagger.CheckResult

_api_return_types = dagger.CheckEntrypointReturnType

_API_RETURN_TYPES = {
    str: _api_return_types.CheckEntrypointReturnString,
    NoneType: _api_return_types.CheckEntrypointReturnVoid,
    dagger.Check: _api_return_types.CheckEntrypointReturnCheck,
    dagger.CheckResult: _api_return_types.CheckEntrypointReturnCheckResult,
}


class CheckResolver(Resolver[CheckReturnType]):
    @override
    def register(self, env: dagger.Environment) -> dagger.Environment:
        check = dagger.check().with_name(self.graphql_name)

        if self.description:
            check = check.with_description(self.description)

        try:
            return_type = _API_RETURN_TYPES[self.return_type]
        except KeyError as e:
            msg = f"Unhandled return type “{self.return_type}”"
            raise TypeError(msg) from e
        return env.with_check(check, return_type=return_type)
