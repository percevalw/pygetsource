import ast
import inspect
import textwrap
from types import CodeType

from typing_extensions import Literal

from pygetsource.ast_utils import reconstruct_arguments_str
from pygetsource.decompiler import getsource


def inspect_function_code(code: CodeType):
    source_code = inspect.getsource(code)
    tree = ast.parse(textwrap.dedent(source_code))
    body = tree.body[0]
    lines = source_code.splitlines()
    body_lines = lines[body.body[0].lineno - 1 :]
    if (
        "\n".join(body_lines).count('"""') % 2 == 1 and body_lines[0].strip() == '"""'
    ) or (
        "\n".join(body_lines).count("'''") % 2 == 1 and body_lines[0].strip() == "'''"
    ):
        body_lines = body_lines[1:]

    function_body = textwrap.dedent("\n".join(body_lines))
    def_str = "async def" if isinstance(body, ast.AsyncFunctionDef) else "def"
    function_name = body.name

    function_code = (
        f"{def_str} {function_name}({reconstruct_arguments_str(code)}):\n"
        + f"{textwrap.indent(function_body, '  ')}\n"
    )
    return function_name, function_code


def getfactory(
    code: CodeType,
    strategy: Literal["inspect", "decompile", "auto"] = "auto",
    function_name: str = "_fn_",
) -> str:
    """
    Get the source code for a factory function to rebuild an equivalent code object
    with closures, linked globals, etc.

    Parameters
    ----------
    code: CodeType
        The code object to decompile
    function_name: str
        The name of the function to create
    strategy: Literal["inspect", "decompile", "auto"]
        The strategy to use to get the source code. If "auto", will try to use
        "inspect" first, and if it fails, will use "decompile".

    Examples
    --------

    Use as follows:

    ```python
    source_code = getfactory(code_object)
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    ```

    Returns
    -------
    str
        The string for the function to evaluate and execute
    """
    function_name = function_code = None
    if strategy in ("auto", "inspect"):
        try:
            function_name, function_code = inspect_function_code(code)
        except Exception:
            if strategy == "inspect":
                raise ValueError()
            else:
                print(
                    f"Could not inspect code of {code.co_name} at "
                    f"{code.co_filename}, trying decompile..."
                )

    if function_name is None:
        function_name = "_fn_"
        function_code = getsource(code, as_function=function_name)

    # Simulate the existence of free variables if there are closures
    # to force Python to insert the right bytecode instructions for loading them.
    if code.co_freevars:
        free_vars = " = ".join(code.co_freevars) + " = None"
        factory_code = (
            "def _factory_():\n"
            + (f"  {free_vars}\n" if free_vars else "")
            + f"{textwrap.indent(function_code, '  ')}\n"
            f"  return {function_name}\n"
            f"_fn_ = _factory_()\n"
        )
    else:
        factory_code = f"{function_code}\n" f"_fn_ = {function_name}\n"

    return factory_code
