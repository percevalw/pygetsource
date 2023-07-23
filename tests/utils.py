import dis
import sys
from types import CodeType
from typing import Callable

from pygetsource import Node
from pygetsource.factory import get_factory_code
from pygetsource.utils import hasjabs


def prune_and_compare(code1: CodeType, code2: CodeType):
    root1 = Node.from_code(code1, prune=True)
    root2 = Node.from_code(code2, prune=True)
    seen = set()
    queue = [(root1, root2)]

    while queue:
        node1, node2 = queue.pop(0)

        if (node1, node2) in seen:
            continue

        seen.add((node1, node2))

        if (node1 is None) != (node2 is None):
            print("(node1 is None) != (node2 is None)", node1, node2)
            return False
        if not node1:
            continue
        if node1.opname != node2.opname:
            print("node1.opname != node2.opname", node1.opname, node2.opname)
            return False

        queue.append((node1.next, node2.next))
        if len(node1.jumps) != len(node2.jumps):
            print("len(node1.jumps) != len(node2.jumps)", len(node1.jumps), len(node2.jumps))
            return False
        if not node1.jumps:
            continue
        queue.append((next(iter(node1.jumps)), next(iter(node2.jumps))))
    return True


def make_test_idem(func):
    def test():
        source_code = get_factory_code(func.__code__)
        res = {}
        print("SOURCE")
        print(source_code)
        exec(source_code, res, res)
        recompiled: Callable = res["_fn_"]
        tgt_codes = func.__code__, *(c for c in func.__code__.co_consts if isinstance(c, CodeType))
        gen_codes = recompiled.__code__, *(c for c in recompiled.__code__.co_consts if isinstance(c, CodeType))
        assert len(tgt_codes) == len(gen_codes)
        for tgt_code, gen_code in zip(gen_codes, tgt_codes):
            assert prune_and_compare(gen_code, tgt_code)
            # nice_gen = nicer_bytecode(remove_nop_codes(gen_code.co_code))
            # nice_tgt = nicer_bytecode(remove_nop_codes(tgt_code.co_code))
            # assert nice_gen == nice_tgt, nice_gen + "\n-----------------\n" + nice_tgt
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
