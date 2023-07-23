import inspect
import textwrap
from types import CodeType

from pygetsource.decompiler import getsource


def reconstruct_arguments_str(code):
    arg_details = inspect.getargs(code)
    sig = []
    if len(arg_details.args):
        sig.append(", ".join(arg_details.args))
    keyword_only_args = getattr(arg_details, "kwonlyargs", None)
    if keyword_only_args:
        sig.append("*, " + ", ".join(keyword_only_args))
    if arg_details.varargs:
        sig.append("*" + arg_details.varargs)
    if arg_details.varkw:
        sig.append("**" + arg_details.varkw)
    return ", ".join(sig)


def getfactory(code: CodeType) -> str:
    """
    Get the source code for a factory function to rebuild an equivalent code object
    with closures, linked globals, etc.

    Parameters
    ----------
    code: CodeType
        The code object to decompile

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
    function_body = getsource(code)

    function_name = "_fn_"
    def_str = "def"

    function_code = (
        f"{def_str} {function_name}({reconstruct_arguments_str(code)}):\n"
        + f"{textwrap.indent(function_body, '  ')}\n"
    )

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
