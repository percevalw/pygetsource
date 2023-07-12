import dis
import functools
import sys
from types import CodeType
from typing import Callable

from pydecompile.factory import get_factory_code
from pydecompile.utils import hasjabs


def make_test_idem(func):
    def test():
        source_code = get_factory_code(func.__code__)
        res = {}
        exec(source_code, res, res)
        recompiled: Callable = res["_fn_"]
        tgt_codes = func.__code__, *(c for c in func.__code__.co_consts if isinstance(c, CodeType))
        gen_codes = recompiled.__code__, *(c for c in recompiled.__code__.co_consts if isinstance(c, CodeType))
        assert len(tgt_codes) == len(gen_codes)
        for tgt_code, gen_code in zip(gen_codes, tgt_codes):
            assert nicer_bytecode(remove_nop_codes(gen_code.co_code)) == nicer_bytecode(remove_nop_codes(tgt_code.co_code)), gen_code
    return test


def remove_nop_codes(bytecode):
    NOP = dis.opmap['NOP']
    shifts = [0] * (len(bytecode) // 2)
    res = []
    for i, (op, arg) in enumerate(zip(bytecode[::2], bytecode[1::2])):
        if sys.version_info < (3, 10) and dis.opname[op] in hasjabs:
            arg //= 2

        if op == NOP:
            for j in range(i + 1, len(shifts)):
                shifts[j] = (shifts[j - 1] - 1) if j > 0 else 0
        else:
            for j in range(i + 1, len(shifts)):
                shifts[j] = shifts[j - 1] if j > 0 else 0

    for i, (op, arg) in enumerate(zip(bytecode[::2], bytecode[1::2])):
        opname = dis.opname[op]
        if sys.version_info < (3, 10) and opname in hasjabs:
            arg //= 2
        if op != NOP:
            if opname in ('JUMP_FORWARD', 'FOR_ITER', 'SETUP_FINALLY'):
                arg += (shifts[i + arg] - shifts[i])
            elif 'JUMP' in opname:
                arg += shifts[arg]

            if sys.version_info < (3, 10) and opname in hasjabs:
                arg *= 2
            res.extend([op.to_bytes(1, 'big'), arg.to_bytes(1, 'big')])
    return b"".join(res)

def nicer_bytecode(x: bytes):
    s = ""
    for code, arg in zip(x[::2], x[1::2]):
        s += f"{dis.opname[code]} {arg}\n"
    return s
