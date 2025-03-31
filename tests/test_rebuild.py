"""
Test the reconstruction of a function from its bytecode.
Although we execute this rebuilt function, it is not the main purpose of the
getfactory, since the function is likely to be missing defaults and
other Function attributes.
"""
from typing import Callable

import pytest

from pygetsource.factory import getfactory


def test_simple():
    def func(a, b):
        return a + b

    source_code = getfactory(func.__code__, strategy="decompile")
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(1, 2) == 3


def test_kw_only_args():
    def func(a, *, b=1):
        return a + b

    source_code = getfactory(func.__code__, strategy="decompile")
    print("source_code", source_code)
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(1, 2) == 3


def test_kwargs():
    def func(a, **d):
        return a + sum(d.values())

    source_code = getfactory(func.__code__, strategy="decompile")
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(a=1, b=2, c=2) == 5


def test_simple_inspect():
    def func(a, b):
        return a + b

    source_code = getfactory(func.__code__, strategy="inspect")
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(1, 2) == 3


def test_lambda_auto():
    func = lambda a, b: a + b  # noqa: E731

    source_code = getfactory(func.__code__, strategy="auto")
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(1, 2) == 3


def test_lambda_inspect_fail():
    func = lambda a, b: a + b  # noqa: E731

    with pytest.raises(ValueError):
        getfactory(func.__code__, strategy="inspect")


def test_inspect_docstring():
    def func(a, b):
        """
        My docstring
        """
        return a + b

    source_code = getfactory(func.__code__, strategy="inspect")
    res = {}
    exec(source_code, res, res)
    recompiled: Callable = res["_fn_"]
    assert recompiled(1, 2) == 3
