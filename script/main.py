# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "dagger-io",
# ]
#
# [tool.uv.sources]
# dagger-io = { path = "sdk", editable = true }
# ///
import dagger
from dagger.mod import run


@dagger.object_type
class Script:
    @dagger.function
    def echo(self, msg: str) -> str:
        return msg


if __name__ == "__main__":
    run(Script)
