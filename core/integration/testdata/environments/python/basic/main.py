import dagger
from dagger.ext import check


def _check_output(name: str) -> str:
    return f"WE ARE RUNNING CHECK {name.replace('_', '-')}"


def _container_check(name: str, succeed: bool) -> dagger.Check:
    cmd = "true" if succeed else "false"
    ctr = (
        dagger.container()
        .from_("alpine:3.18")
        .with_exec(["sh", "-e", "-c", f"echo {_check_output(name)}; {cmd}"])
    )
    return dagger.check().with_name(name).with_container(ctr)


@check
def cool_static_check() -> dagger.CheckResult:
    return dagger.static_check_result(True, output=_check_output("cool_static_check"))


@check
def sad_static_check() -> dagger.CheckResult:
    return dagger.static_check_result(False, output=_check_output("sad_static_check"))


@check
def cool_container_check() -> dagger.Check:
    return _container_check("cool_container_check", True)


@check
def sad_container_check() -> dagger.Check:
    return _container_check("sad_container_check", False)


@check
def cool_string_only_return() -> str:
    return _check_output("cool_string_only_return")


@check
def cool_error_only_return() -> None:
    return


@check
def sad_error_only_return() -> None:
    msg = _check_output("sad_error_only_return")
    raise ValueError(msg)
