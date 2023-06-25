import ast
import dis
import sys
from collections import defaultdict
from types import CodeType
from typing import List, Tuple, Optional

import astunparse

binop_to_ast = {
    dis.opmap["BINARY_MATRIX_MULTIPLY"]: ast.MatMult(),
    dis.opmap["BINARY_POWER"]: ast.Pow(),
    dis.opmap["BINARY_MULTIPLY"]: ast.Mult(),
    dis.opmap["BINARY_MODULO"]: ast.Mod(),
    dis.opmap["BINARY_ADD"]: ast.Add(),
    dis.opmap["BINARY_SUBTRACT"]: ast.Sub(),
    dis.opmap["BINARY_FLOOR_DIVIDE"]: ast.FloorDiv(),
    dis.opmap["BINARY_TRUE_DIVIDE"]: ast.Div(),
    dis.opmap["BINARY_LSHIFT"]: ast.LShift(),
    dis.opmap["BINARY_RSHIFT"]: ast.RShift(),
    dis.opmap["BINARY_AND"]: ast.BitAnd(),
    dis.opmap["BINARY_XOR"]: ast.BitXor(),
    dis.opmap["BINARY_OR"]: ast.BitOr(),
}

unaryop_to_ast = {
    dis.opmap["UNARY_POSITIVE"]: ast.UAdd(),
    dis.opmap["UNARY_NEGATIVE"]: ast.USub(),
    dis.opmap["UNARY_NOT"]: ast.Not(),
    dis.opmap["UNARY_INVERT"]: ast.Invert(),
}

inplace_to_ast = {
    dis.opmap["INPLACE_MATRIX_MULTIPLY"]: ast.MatMult(),
    dis.opmap["INPLACE_FLOOR_DIVIDE"]: ast.FloorDiv(),
    dis.opmap["INPLACE_TRUE_DIVIDE"]: ast.Div(),
    dis.opmap["INPLACE_ADD"]: ast.Add(),
    dis.opmap["INPLACE_SUBTRACT"]: ast.Sub(),
    dis.opmap["INPLACE_MULTIPLY"]: ast.Mult(),
    dis.opmap["INPLACE_MODULO"]: ast.Mod(),
    dis.opmap["INPLACE_POWER"]: ast.Pow(),
    dis.opmap["INPLACE_LSHIFT"]: ast.LShift(),
    dis.opmap["INPLACE_RSHIFT"]: ast.RShift(),
    dis.opmap["INPLACE_AND"]: ast.BitAnd(),
    dis.opmap["INPLACE_XOR"]: ast.BitXor(),
    dis.opmap["INPLACE_OR"]: ast.BitOr(),
}

cmp_map = {op: i for i, op in enumerate(dis.cmp_op)}

compareop_to_ast = {
    cmp_map["=="]: ast.Eq(),
    cmp_map["<"]: ast.Lt(),
    cmp_map["<="]: ast.LtE(),
    cmp_map[">"]: ast.Gt(),
    cmp_map[">="]: ast.GtE(),
    cmp_map["!="]: ast.NotEq(),
}
if sys.version_info < (3, 9):
    compareop_to_ast[cmp_map["exception match"]] = "exception match"
    compareop_to_ast[cmp_map["in"]] = ast.In()
    compareop_to_ast[cmp_map["not in"]] = ast.NotIn()
    compareop_to_ast[cmp_map["is"]] = ast.Is()
    compareop_to_ast[cmp_map["is not"]] = ast.IsNot()



def postprocess_try_finally(node):
    if len(node.body) == 1 and isinstance(node.body[0], ast.Try) and node.finalbody:
        node.handlers = node.body[0].handlers
        node.orelse = node.body[0].orelse
        node.body = node.body[0].body
    else:
        try:
            if not node.body:
                node.body = [ast.Pass()]
            for handler in node.handlers:
                if len(handler.body) == 1 and isinstance(handler.body[0], ast.Try):
                    child = handler.body[0]

                    # Lookup "except: e = None; del e" in the handler children
                    assert isinstance(child, ast.Try)
                    if len(child.finalbody) == 2:
                        fb0 = child.finalbody[0]
                        fb1 = child.finalbody[1]
                    elif len(child.handlers) == 1:
                        fb0 = child.handlers[0].body[0]
                        fb1 = child.handlers[0].body[1]
                    else:
                        continue

                    if (
                        isinstance(fb0, ast.Assign)
                        and fb0.targets[0].id == handler.name
                        and fb0.value.value is None
                        and isinstance(fb1, ast.Delete)
                        and fb1.targets[0].id == handler.name
                    ):
                        # If found, replace the body of the handler with the body of
                        # the child
                        handler.body = child.body or [ast.Pass()]
                        if (
                            isinstance(fb0, ast.Assign)
                            and fb0.targets[0].id == handler.name
                            and fb0.value.value is None
                            and isinstance(fb1, ast.Delete)
                            and fb1.targets[0].id == handler.name
                        ):
                            handler.body = child.body or [ast.Pass()]

        finally:
            pass
    return node


def postprocess_loop(node):
    assert isinstance(node, (ast.While, ast.For))
    # Remove continue at the end of the loop
    # node.body = node.body[:-1]
    if len(node.body) > 0:
        if isinstance(node.body[-1], ast.Try):
            child = node.body[-1]
            if child.body and isinstance(child.body[-1], ast.Continue):
                child.body = child.body[:-1]
            if child.orelse and isinstance(child.orelse[-1], ast.Continue):
                child.orelse = child.orelse[:-1]
            for handler in child.handlers:
                if handler.body and isinstance(handler.body[-1], ast.Continue):
                    handler.body = handler.body[:-1]
        for i, if_node in list(enumerate(node.body))[::-1]:
            if isinstance(if_node, ast.If):
                if (
                    len(if_node.body)
                    and isinstance(if_node.body[-1], ast.Continue)
                    and not if_node.orelse
                ):
                    if_node.body = if_node.body[:-1] or [ast.Continue()]
                    if_node.orelse = node.body[i + 1 :] or None
                    node.body = node.body[:i] + [if_node]
    return node


def extract_yield(node):
    assert isinstance(node, ast.Expr), f"Expected Expr instead of {ast.dump(node)}"
    node = node.value
    assert isinstance(node, ast.Yield), f"Expected Yield instead of {ast.dump(node)}"
    return node.value


def split_stack(stack, from_idx):
    rest = []
    res = []
    for idx, op in stack:
        if idx >= from_idx:
            res.append((idx, op))
        else:
            rest.append((idx, op))
    return rest, res


def process_jumps(bytecode):
    loops_end = {}
    jump_origins = defaultdict(list)
    for idx in range(0, len(bytecode), 2):
        op, arg = bytecode[idx], bytecode[idx + 1]
        opname = dis.opname[op]
        if opname in hasjabs:
            arg *= 2
        if opname in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
            jump_origins[arg].append(idx)
            if arg <= idx:
                loops_end[arg] = max(idx, loops_end.get(arg, idx))
        elif opname in ("JUMP_ABSOLUTE",):
            jump_origins[arg].append(idx)
            if arg <= idx:
                loops_end[arg] = max(idx, loops_end.get(arg, idx))
        elif opname in ("JUMP_FORWARD",):
            jump_origins[idx + arg + 2].append(idx)

    # Should we perform graph completion for the jumps ?
    return {k: v + 2 for k, v in loops_end.items()}, jump_origins


def extract_items(args):
    items = []
    for x in args:
        if isinstance(x, (ast.Tuple, ast.List, ast.Set)):
            items.extend(x.elts)
        elif isinstance(x, ast.Constant):
            for const in x.value:
                items.append(ast.Constant(const, kind=None))
        else:
            items.append(ast.Starred(x))
    return items


class Unpacking:
    def __init__(self, value, counter, starred=False):
        self.value = value
        self.counter = counter
        self.starred = starred

    def __repr__(self):
        return f"Unpacking({self.value}, {self.counter}, starred={self.starred})"


hasjabs = ["JUMP_ABSOLUTE", "FOR_ITER", "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"]


def build_ast(code: CodeType, as_function=True, version: Optional[Tuple[int, int]] = None):
    if version is None:
        version = sys.version_info
    bytecode = code.co_code
    offset = 0

    def read(n=-1, batched: Optional[int] = None):
        nonlocal offset
        total = 0
        while total != n:
            if offset >= len(bytecode):
                return
            batch = []
            for i in range(batched or 1):
                item = bytecode[offset]
                total += 1
                offset += 1
                batch.append(item)
            if batched:
                yield batch
            else:
                yield item  # , ctx=ast.Load())

    def UnknownException():
        bytecode_instructions = list(dis.Bytecode(code))
        bytecode_str = "\n".join(str(x) for x in bytecode_instructions)
        return Exception(
            f"Unsupported {opname} ({arg}) at {op_idx}:\n{bytecode_str}\nCurrent ctx: {ctx_stack}\nCurrent stack: {stack}"
        )

    def add_instruction(inst, start: int):
        # nonlocal next_instruction_idx

        # This is for list comphrensions
        # where we "(return (load the_initial_built_list))
        # Since we usually take the idx of the load instruction
        # for the return/store/etc instructions, and the load instruction
        # is high up in the stack, this would break the order of the instructions
        # So we ensure that the start of the new instruction is always after the
        # last instruction
        start = max((start, *(i[0] for i in instructions)))
        instructions.append((start, inst))
        print(" > INST", start, ast.dump(inst) if isinstance(inst, ast.AST) else None)
        # op_idx = offset

    def add_stack(node, start):
        stack.append((op_idx if start is None else start, node))

    def unzip_stack(n):
        elements = [stack.pop() for _ in range(n)][::-1]
        return zip(*elements) if len(elements) else ((), ())

    def process_binding(opname, arg):
        opname = opname.split("_")[1]
        if opname == "FAST":
            return None, ast.Name(code.co_varnames[arg])

        elif opname in ("NAME", "GLOBAL"):
            return None, ast.Name(code.co_names[arg])

        elif opname == "DEREF":
            return None, ast.Name((code.co_cellvars + code.co_freevars)[arg])
        elif opname == "CONST":
            if (
                dis.opname[bytecode[offset]] == "SETUP_FINALLY"
                and code.co_consts[arg] is None
            ):
                pass
            else:
                return None, ast.Constant(code.co_consts[arg], kind=None)
        # ATTRIBUTES
        elif opname in ("ATTR", "METHOD"):
            idx, value = stack.pop()
            return idx, ast.Attribute(value=value, attr=code.co_names[arg])
        elif opname in ("SUBSCR",):
            idx, value = stack.pop()
            return idx, ast.Subscript(
                slice=value,
                value=stack.pop()[1],
            )
        return None, None

    def dump_debug(x):
        try:
            return ast.dump(x)
        except:
            return x

    def process_jump_absolute(opname, arg):
        loops = [
            ctx
            for ctx in ctx_stack
            if ctx["kind"] == "ctrl" and ctx["start"] in loop_ends
               and ("else" not in ctx or ctx["else"] == loop_ends[ctx["start"]])
        ]
        ctrls = [ctx for ctx in ctx_stack if ctx["kind"] == "ctrl"]
        print("CTRLS", loops, arg)
        if len(loops) > 0:
            current_start = loops[-1]["start"]
            current_end = loop_ends.get(current_start)
        else:
            current_end = current_start = None
        if len(loops) > 1:
            upper_start = loops[-2]["start"]
        else:
            upper_start = None

        if arg == current_start:
            matching_loops = [
                ctx
                for ctx in loops[::-1]
                if ctx["start"] == current_start
                   and ctx["else"] == current_end
            ]
            if matching_loops and not any(ctx["loop"] for ctx in matching_loops):
                matching_loops[-1]["loop"] = True
            # loop = next((
            #    ctx
            #    for ctx in loops[::-1]
            #    if ctx["start"] == current_start
            #     and ctx["else"] == current_end
            #     and not ctx["loop"]
            # ), None)
            # if loop:
            #    loop["loop"] = True
            # Don't insert a continue stmt if we're about to end the loop's body
            if not ("else" in loops[-1] and loops[-1]["else"] == offset):
                add_instruction(ast.Continue(), start=op_idx)
            else:
                print("Not adding continue stmt because we're at the end of the loop")
        elif arg == current_end:
            loops[-1]["loop"] = True
            add_instruction(ast.Break(), start=op_idx)
        elif arg == upper_start:
            loops[-2]["loop"] = True
            add_instruction(ast.Break(), start=op_idx)
        elif arg in loop_ends.keys():
            # If we're going backward to a position not known as a structure
            # start, this must be a new infinite loop
            print("Adding infinite loop due to likely continue ?")
            ctx = {
                "kind": "ctrl",
                "body_start": op_idx + 2,
                "start": arg,
                "else": loop_ends[arg],
                "is_for": False,
                "loop": True,
                "has_test": False,
            }
            ctx_stack.append(ctx)

            if not ("else" in ctx and ctx["else"] == offset):
                add_instruction(ast.Continue(), start=op_idx)
        elif arg > op_idx and len(ctrls):
            print("Adding infinite loop due to likely break")
            # idx_else = min((arg, *(ctrls[-2]["else"] if len(ctrls) > 1 else 0,)))
            idx_else = arg
            idx_start = ctrls[-1]["start"]

            # Look up jump statement after which we could end the while loop
            candidate_ends = jump_origins.get(arg, []) + [arg]
            print("CANDIDATE ENDS", candidate_ends)
            for candidate_end in candidate_ends:
                # If the jump is after the current instruction, and we are garanteed
                # that the while will never loop, because of the presence of jump/return stmt
                if candidate_end > op_idx and dis.opname[bytecode[candidate_end - 2]] in ("RETURN_VALUE", "JUMP_FORWARD", "JUMP_ABSOLUTE"):
                    print(" > Could add while at", candidate_end)
                    idx_else = candidate_end
                    break
            else:
                raise Exception("Could not find a suitable end for the while loop")

            # Now we just have to find a suitable start for the while loop
            # such that it doesn't cross any if/else block
            for ctx in reversed(ctrls):
                if ctx["start"] <= idx_else and ctx.get("end", ctx.get("else")) <= idx_else:
                    print("Found block", ctx)
                    idx_start = ctx["start"]
                    # idx_else = arg#ctx.get("end", ctx.get("else"))
                    break

            # Fix if/else crossing statements
            # If an if/else block starts before but has only an else that occurs
            # before the break, it's time to define an end for this block
            # See step-by-step-2.txt for an example
            # for ctx in ctx_stack[::-1]:
            #     if (
            #           ctx["kind"] == "ctrl"
            #           and ctx["start"] < idx_start
            #           and ctx.get("end", ctx.get("else")) < idx_else
            #     ):
            #         print(f"Fixing if/else crossing statements: {ctx} / {idx_start}->{ctx['else'] - 2} instead of {idx_start}->{idx_else}")
            #         idx_else = ctx["else"] - 2
            #         break

            ctx = {
                "kind": "ctrl",
                "body_start": op_idx + 2,
                "start": idx_start,
                "else": idx_else,  # decide between -2 or -0
                "is_for": False,
                "loop": True,
                "has_test": False,
            }
            add_instruction(ast.Break(), start=op_idx)
            ctx_stack.append(ctx)
            # ctrls = [ctx for ctx in ctx_stack if ctx["kind"] == "ctrl"]
            # if len(ctrls) > 0:
            #    ctrls[-1]["loop"] = True
            #    add_instruction(ast.Break(), start=op_idx)
            # else:
            #    print(loops)
            #    raise UnknownException()
        else:
            raise UnknownException()

    ctx_stack = []
    instructions: List[Tuple[int, ast.AST]] = []
    stack = []
    op_idx = 0

    # Detect loops beforehand
    loop_ends, jump_origins = process_jumps(bytecode)
    print("LOOP ENDS", loop_ends)
    print("JUMP ORIGINS", jump_origins)

    try:
        for op, arg in read(batched=2):
            opname = dis.opname[op]
            op_idx = offset - 2

            # Argument byte prefixing for too large arguments
            # https://docs.python.org/3.8/library/dis.html#opcode-EXTENDED_ARG
            ex_arg_count = 0
            ex_arg = 0
            while opname == "EXTENDED_ARG":
                ex_arg = arg << (8 * (ex_arg_count + 1))
                [op, arg] = read(2)
                opname = dis.opname[op]
                op_idx = offset - 2
            arg += ex_arg

            if version >= (3, 10) and opname in hasjabs:
                arg *= 2

            # ==================== #

            ctx_stack = sorted(
                ctx_stack,
                # Process blocks that start first and end last before others
                key=lambda ctx: (
                    ctx["start"],
                    -ctx.get("end", ctx.get("else", ctx.get("end_body"))),
                    not ctx.get("loop", False)
                ),
            )
            print("CODE", str(op_idx).ljust(3), opname.ljust(18), str(arg).ljust(3), [(i, dump_debug(x)) for i, x in stack])
            for ctx in ctx_stack:
               print("   ctx>", ctx)
            new_ctx_stack = list(ctx_stack)
            for ctx in ctx_stack[::-1]:
                if ctx["kind"] == "try" and "end" in ctx and ctx.get("end") <= op_idx:
                    body_block = [
                        inst
                        for idx, inst in instructions
                        if ctx["start"] <= idx < ctx["end_body"]
                    ]
                    if "else" in ctx:
                        else_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["else"] <= idx < ctx["end"]
                        ]
                    else:
                        else_block = []
                    if len(ctx["except"]):
                        except_blocks = [
                            [inst for idx, inst in instructions if start <= idx < end]
                            for start, end in zip(
                                ctx["except"],
                                ctx["except"][1:] + [ctx.get("else", ctx.get("end"))],
                            )
                        ]
                    else:
                        except_blocks = []
                    if ctx.get("begin_finally") is not None:
                        finally_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["begin_finally"] <= idx < ctx["end"]
                        ]
                    else:
                        finally_block = []
                    if not len(finally_block) and not len(except_blocks):
                        except_blocks = [None]
                    instructions = [i for i in instructions if i[0] < ctx["start"]]
                    add_instruction(
                        postprocess_try_finally(
                            ast.Try(
                                body=body_block,
                                handlers=[
                                    ast.ExceptHandler(
                                        type=ctx["except_types"].get(i)[0]
                                        if i in ctx["except_types"]
                                        else None,
                                        name=ctx["except_types"].get(i)[1]
                                        if i in ctx["except_types"]
                                        else None,
                                        body=except_block or [ast.Pass()],
                                    )
                                    for i, except_block in enumerate(except_blocks)
                                ],
                                orelse=else_block or None,
                                finalbody=finally_block,
                            )
                        ),
                        start=ctx["start"],
                    )
                elif ctx["kind"] == "ctrl" and (
                    "end" in ctx
                    and ctx["end"] <= op_idx
                    or "end" not in ctx
                    and "else" in ctx
                    and ctx["else"] <= op_idx
                ):
                    # print("DOING ctrl", op_idx, ctx)
                    # for i, inst in instructions:
                    #    print(" -  ", i, dump_debug(inst))

                    if ctx["is_for"]:
                        iterator = next(
                            inst for idx, inst in instructions if ctx["start"] == idx
                        )
                        target_idx, target = next(
                            (idx, inst)
                            for idx, inst in instructions
                            if idx > ctx["start"]
                        )
                        body_block = [
                            inst
                            for idx, inst in instructions
                            if target_idx < idx < ctx["else"]
                        ] or [ast.Pass()]
                        else_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["else"] <= idx <= ctx["end"]
                        ]
                        if isinstance(target, ast.Assign):
                            target = target.targets[0]
                        instructions = [i for i in instructions if i[0] < ctx["start"]]
                        add_instruction(
                            postprocess_loop(
                                ast.For(
                                    # (ast.For(
                                    iter=iterator,
                                    target=target,
                                    body=body_block,
                                    orelse=else_block or None,
                                )
                            ),
                            start=ctx["start"],
                        )
                    else:
                        # print("IF BLOCK", ctx)
                        if ctx["has_test"]:
                            body_start = ctx["start"] + 1
                            condition = next(
                                inst
                                for idx, inst in instructions
                                if idx == ctx["start"]
                            )
                        else:
                            body_start = ctx["start"]
                            condition = ast.Constant(True, kind=None)

                        is_ternary = False
                        body_block = [
                            inst
                            for idx, inst in instructions
                            if body_start <= idx < ctx["else"]
                        ]
                        if not ctx["loop"] and not len(body_block):
                            assert ctx[
                                "has_test"
                            ], "Weird, this looks like a ternary but body/ctrl could not be found"
                            stack_block = [
                                inst
                                for idx, inst in stack
                                if body_start <= idx < ctx["else"]
                            ]
                            if len(stack_block) == 1:
                                is_ternary = True
                                body_block = stack_block
                            else:
                                print("WARNING: Could not find body block for if/else")
                                #raise UnknownException()
                                body_block = [ast.Continue()]#ast.Continue()]
                        else:
                            body_block = body_block or [ast.Pass()]

                        if "end" in ctx:
                            else_block = [
                                inst
                                for idx, inst in instructions
                                if ctx["else"] <= idx <= ctx["end"]
                            ]

                            if not len(else_block):
                                else_stack_block = [
                                    inst
                                    for idx, inst in stack
                                    if ctx["else"] <= idx <= ctx["end"]
                                ]
                                if len(else_stack_block) == 1:
                                    is_ternary = True
                                    else_block = else_stack_block
                                else:
                                    else_block = []
                        else:
                            else_block = []

                        if is_ternary:
                            stack = [i for i in stack if i[0] < ctx["start"]]
                        instructions = [i for i in instructions if i[0] < ctx["start"]]

                        if ctx.get("loop"):
                            add_instruction(
                                postprocess_loop(
                                    ast.While(
                                        # (ast.While(
                                        test=condition,
                                        body=body_block,
                                        orelse=else_block or None,
                                    )
                                ),
                                start=ctx["start"],
                            )
                        elif not is_ternary:
                            add_instruction(
                                ast.If(
                                    test=condition,
                                    body=body_block,
                                    orelse=else_block or None,
                                ),
                                start=ctx["start"],
                            )
                        else:
                            add_stack(
                                ast.IfExp(
                                    test=condition,
                                    body=body_block,
                                    orelse=else_block or None,
                                ),
                                ctx["start"],
                            )
                            print("::: AS TERNARY", ast.dump(stack[-1][1]))
                else:
                    # new_ctx_stack.append(ctx)
                    break
                new_ctx_stack.remove(ctx)

            ctx_stack = new_ctx_stack  # [::-1]

            ##################################
            # ASSIGNMENTS

            while True:
                if opname.startswith("LOAD_"):
                    idx, value = process_binding(opname, arg)
                    add_stack(value, start=idx or op_idx)
                elif opname.startswith(
                    "STORE_",
                ):
                    target = process_binding(opname, arg)[1]
                    idx, value = stack.pop()
                    if isinstance(value, ast.AugAssign):
                        add_instruction(value, start=op_idx)
                    elif isinstance(value, ast.FunctionDef) and isinstance(
                        target, ast.Name
                    ):
                        add_instruction(value, start=op_idx)
                    else:
                        unpacking = value if isinstance(value, Unpacking) else None
                        if unpacking:
                            value = value.value
                            target = ast.Starred(target) if unpacking.starred else target
                            multi_targets = [ast.Tuple([target])]
                        else:
                            multi_targets = [target]

                        if (
                            len(instructions)
                            and isinstance(instructions[-1][1], ast.Assign)
                            and idx <= instructions[-1][0]
                        ):
                            prev = instructions.pop()[1]
                            prev_multi_targets = prev.targets

                            if (
                                (not unpacking and prev.value is value)
                                or (
                                    unpacking
                                    and (
                                        len(prev_multi_targets) == 0
                                        or not isinstance(prev_multi_targets[-1], ast.Tuple)
                                    )
                                )
                                or (
                                    unpacking
                                    and len(prev_multi_targets[-1].elts) > unpacking.counter
                                )
                            ):
                                multi_targets = prev_multi_targets + multi_targets
                            else:
                                if len(prev_multi_targets) > 0 and isinstance(
                                    prev_multi_targets[-1], ast.Tuple
                                ):
                                    multi_targets = [
                                        *prev_multi_targets[:-1],
                                        ast.Tuple(prev_multi_targets[-1].elts + [target]),
                                    ]
                                else:
                                    multi_targets = [
                                        *prev_multi_targets[:-1],
                                        ast.Tuple([prev_multi_targets[-1], target]),
                                    ]
                                if not unpacking:
                                    if isinstance(prev.value, ast.Tuple):
                                        value = ast.Tuple(
                                            [*prev.value.elts, value], ctx=ast.Load()
                                        )
                                    else:
                                        value = ast.Tuple(
                                            [prev.value, value], ctx=ast.Load()
                                        )
                                # value is value

                        add_instruction(
                            ast.Assign(
                                targets=multi_targets,
                                value=value,
                            ),
                            start=op_idx,
                        )
                elif opname.startswith("DELETE_"):
                    idx, value = process_binding(opname, arg)
                    add_instruction(ast.Delete([value]), start=op_idx)
                elif (
                    opname in ("LIST_APPEND", "SET_ADD") and ctx_stack[-1]["kind"] == "ctrl"
                ):
                    # idx, collection = stack[-arg]
                    # print("STACK", stack)
                    # print("COLLECTION", collection)
                    idx, value = stack.pop()
                    add_instruction(ast.Expr(ast.Yield(value)), start=idx)
                elif opname == "MAP_ADD" and ctx_stack[-1]["kind"] == "ctrl":
                    # assume we're in a for-loop
                    # We're in a comprehension, and don't do anything with the list_idx
                    # index at this point. We only fake a yield to retrieve it later
                    # when the comprehension is called from the parent function.
                    stack, pairs = split_stack(stack, ctx_stack[-1]["start"])
                    for (key_idx, key), (value_idx, value) in zip(
                        pairs[-2::-2], pairs[-1::-2]
                    ):
                        if version < (3, 8):
                            add_instruction(
                                ast.Expr(ast.Yield(value=ast.Tuple([value, key]))),
                                start=key_idx,
                            )
                        else:
                            add_instruction(
                                ast.Expr(ast.Yield(value=ast.Tuple([key, value]))),
                                start=key_idx,
                            )

                # DELETIONS
                elif opname == "UNPACK_SEQUENCE":
                    idx, value = stack.pop()
                    for i in reversed(range(arg)):
                        stack.append((idx, Unpacking(value, counter=i)))
                elif opname == "UNPACK_EX":
                    # FROM Python's doc:
                    # The low byte of counts is the number of values before the list value,
                    # the high byte of counts the number of values after it. The resulting
                    # values are put onto the stack right-to-left.
                    idx, value = stack.pop()
                    before, after = arg & 0xFF, arg >> 8

                    for i in reversed(range(before + after + 1)):
                        stack.append(
                            (idx, Unpacking(value, counter=i, starred=i == before))
                        )

                # BUILD COLLECTIONS
                elif opname == "BUILD_TUPLE":
                    indices, args = unzip_stack(arg)
                    add_stack(ast.Tuple(args), start=indices[0] if indices else op_idx)
                elif opname == "BUILD_LIST":
                    indices, args = unzip_stack(arg)
                    add_stack(ast.List(args), start=indices[0] if indices else op_idx)
                elif opname == "BUILD_SET":
                    indices, args = unzip_stack(arg)
                    add_stack(
                        ast.Set(args)
                        if len(args) >= 1
                        else ast.Call(func=ast.Name("set"), args=args, keywords=[]),
                        start=indices[0] if indices else op_idx,
                    )
                elif opname == "BUILD_MAP":
                    indices, args = unzip_stack(arg * 2)
                    add_stack(
                        ast.Dict(args[::2], args[1::2]),
                        start=indices[0] if indices else op_idx,
                    )
                elif opname == "BUILD_CONST_KEY_MAP":
                    keys = [ast.Constant(key, kind=None) for key in stack.pop()[1].value]
                    indices, values = unzip_stack(arg)
                    add_stack(
                        ast.Dict(keys, values), start=indices[0] if indices else op_idx
                    )
                elif opname in ("BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL"):
                    indices, args = unzip_stack(arg)
                    items = extract_items(args)
                    add_stack(ast.Tuple(items), start=indices[0] if indices else op_idx)
                elif opname == "BUILD_LIST_UNPACK":
                    indices, args = unzip_stack(arg)
                    items = extract_items(args)
                    add_stack(ast.List(items), start=indices[0] if indices else op_idx)
                elif opname == "BUILD_SET_UNPACK":
                    indices, args = unzip_stack(arg)
                    items = extract_items(args)
                    add_stack(ast.Set(items), start=indices[0] if indices else op_idx)
                elif opname in ("BUILD_MAP_UNPACK", "BUILD_MAP_UNPACK_WITH_CALL"):
                    indices, args = unzip_stack(arg)
                    unpacked = []
                    for dic in args:
                        if isinstance(dic, ast.Dict) and all(
                            key.value is not None
                            and isinstance(key.value, str)
                            and key.value.isidentifier()
                            for key in dic.keys
                            if key is not None and isinstance(key, ast.Constant)
                        ):
                            # check for valid python keywords identifiers
                            unpacked.extend(zip(dic.keys, dic.values))
                        else:
                            unpacked.extend([(None, dic)])
                    keys = [k for k, v in unpacked]
                    values = [v for k, v in unpacked]
                    add_stack(
                        ast.Dict(keys, values), start=indices[0] if indices else op_idx
                    )
                # INPLACE/BINARY OPERATIONS
                elif op in binop_to_ast:
                    right = stack.pop()[1]
                    idx, left = stack.pop()
                    add_stack(
                        ast.BinOp(left=left, op=binop_to_ast[op], right=right),
                        start=idx,
                    )
                elif op in inplace_to_ast:
                    right = stack.pop()[1]
                    idx, left = stack.pop()
                    add_stack(
                        ast.AugAssign(target=left, op=inplace_to_ast[op], value=right),
                        start=idx,
                    )
                elif opname == "BINARY_SUBSCR":
                    slice = stack.pop()[1]
                    idx, value = stack.pop()
                    add_stack(ast.Subscript(value, slice), start=idx)
                elif op in unaryop_to_ast:
                    idx, value = stack.pop()
                    add_stack(ast.UnaryOp(op=unaryop_to_ast[op], operand=value), start=idx)

                # FUNCTIONS
                elif opname == "RETURN_VALUE":
                    idx, value = stack.pop()
                    add_instruction(ast.Return(value=value), start=idx)

                    # Check if we're inside a loop, because python
                    # removes the jump_absolute (that we parse as a continue statement)
                    # in these cases
                    loops = [
                        ctx
                        for ctx in ctx_stack
                        if ctx["kind"] == "ctrl" and ctx["start"] in loop_ends
                           and ("else" not in ctx or ctx["else"] == loop_ends[ctx["start"]])
                    ]
                    if len(loops) and not ("else" in loops[-1] and loops[-1]["else"] == offset + 2):
                        add_instruction(ast.Continue(), start=idx)


                elif opname == "CALL_FUNCTION" or opname == "CALL_METHOD":
                    args = [stack.pop()[1] for _ in range(arg)][::-1]
                    idx, func = stack.pop()
                    if isinstance(func, ast.FunctionDef):

                        class RewriteComprehensionArgs(ast.NodeTransformer):
                            def visit_Name(self, node):
                                if node.id.startswith("."):
                                    var_idx = int(node.id[1:])
                                    return args[var_idx]
                                return node

                            def visit_For(self, node: ast.For) -> ast.AST:
                                assert len(node.body) == 1
                                target = self.visit(node.target)
                                elt = self.visit(node.body[0])
                                iter = self.visit(node.iter)
                                condition = None
                                if isinstance(elt, ast.If):
                                    condition = elt.test
                                    assert len(elt.body) == 1
                                    elt = elt.body[0]
                                generators = [
                                    ast.comprehension(
                                        target=target,
                                        iter=iter,
                                        ifs=[condition]
                                        if condition is not None
                                        else [],  # TODO
                                    )
                                ]
                                # if not isinstance(elt, ast.IfExp):
                                #    raise Exception("Expected IfExp instead of " + ast.dump(elt))
                                # TODO handle DictComp
                                if isinstance(elt, cls):
                                    generators = generators + elt.generators
                                    elt = elt.elt
                                else:
                                    elt = extract_yield(elt)

                                if cls is ast.DictComp:
                                    assert isinstance(elt, ast.Tuple)
                                    return cls(
                                        key=elt.elts[0],
                                        value=elt.elts[1],
                                        generators=generators,
                                        ifs=[],
                                    )
                                elif cls is ast.ListComp:
                                    return cls(
                                        elt=elt,
                                        generators=generators,
                                        ifs=[],
                                    )
                                else:  # cls is ast.SetComp:
                                    return cls(
                                        elt=elt,
                                        generators=generators,
                                        ifs=[],
                                    )

                        assert len(func.body) == 2
                        cls = (
                            ast.DictComp
                            if isinstance(func.body[1].value, ast.Dict)
                            else ast.ListComp
                            if isinstance(func.body[1].value, ast.List)
                            else ast.SetComp
                        )
                        tree = RewriteComprehensionArgs().visit(func)

                        if len(func.body) == 1 or isinstance(func.body[1], ast.Return):
                            tree = func.body[0]
                        add_stack(tree, start=idx)
                    else:
                        add_stack(
                            ast.Call(
                                func=func,
                                args=args,
                                keywords=[],
                            ),
                            start=idx,
                        )
                elif opname == "CALL_FUNCTION_KW":
                    keys = stack.pop()[1].value
                    values = [stack.pop()[1] for _ in range(len(keys))][::-1]
                    args = [stack.pop()[1] for _ in range(arg - len(keys))][::-1]
                    idx, func = stack.pop()
                    add_stack(
                        ast.Call(
                            func=func,
                            args=args,
                            keywords=[
                                ast.keyword(arg=key, value=value)
                                for key, value in zip(keys, values)
                            ],
                        ),
                        start=idx,
                    )
                elif opname == "CALL_FUNCTION_EX":
                    if arg & 0x01:
                        kwargs = stack.pop()[1]
                        args = stack.pop()[1]
                    else:
                        kwargs = None
                        args = stack.pop()[1]
                    idx, func = stack.pop()
                    if isinstance(kwargs, ast.Dict):
                        keywords = [
                            ast.keyword(
                                arg=key.value if key is not None else None, value=value
                            )
                            for key, value in zip(kwargs.keys, kwargs.values)
                        ]
                    else:
                        keywords = [ast.keyword(arg=None, value=kwargs)]
                    if hasattr(args, "elts"):
                        args = args.elts
                    elif isinstance(args, ast.Constant) and isinstance(
                        args.value, (tuple, list)
                    ):
                        args = [ast.Constant(value=elt, kind=True) for elt in args.value]
                    add_stack(
                        ast.Call(
                            func=func,
                            args=args,
                            keywords=keywords,
                        ),
                        start=idx,
                    )

                # Control structures
                elif opname in ("SETUP_FINALLY", "SETUP_EXCEPT"):
                    ctx_stack.append(
                        {
                            "start": op_idx,
                            "end_body": op_idx + arg + 2,
                            "except": [],
                            "except_types": {},
                            "kind": "try",
                            "begin_finally": (
                                op_idx + arg + 2
                                if opname == "SETUP_FINALLY" and version < (3, 8)
                                else None
                            )
                        }
                    )
                    if (
                        dis.opname[bytecode[op_idx + arg + 2]] != "END_FINALLY"
                        and dis.opname[bytecode[op_idx + arg]] != "BEGIN_FINALLY"
                        and dis.opname[bytecode[op_idx + arg + 2]] != "POP_FINALLY"
                    ):
                        ctx_stack[-1]["except"].append(op_idx + arg + 2)
                    # elif dis.opname[bytecode[op_idx + arg]] == "BEGIN_FINALLY" or dis.opname[bytecode[op_idx + arg + 2]] == "POP_FINALLY":
                    #    ctx_stack[-1]["begin_finally"] = op_idx + arg
                elif opname == "POP_BLOCK":
                    pass
                elif opname == "POP_TOP":
                    # TODO should we check if we're in a loop ?
                    next_code =  dis.opname[bytecode[offset]] if offset < len(bytecode) else None
                    if next_code is not None and (next_code not in (
                        "JUMP_ABSOLUTE",
                        "RETURN_VALUE",
                    ) or next_code == "JUMP_ABSOLUTE" and bytecode[offset + 1] < offset):
                        if len(stack) > 0:
                            top = stack.pop()[1]
                            if top is not None:
                                # if top is not None and top is not EXC and not isinstance(top, ast.Constant):
                                add_instruction(ast.Expr(top), start=op_idx)

                    # ctx_stack[-1]["end_body"] = next_instruction_idx
                elif opname == "POP_EXCEPT":
                    pass
                elif opname == "POP_FINALLY":
                    pass

                # JUMPS
                elif opname == "JUMP_FORWARD":
                    # try/except without Exception type
                    assert ctx_stack[-1]["kind"] in ("try", "ctrl")
                    if ctx_stack[-1]["kind"] == "try":
                        if ctx_stack[-1]["except"] and op_idx > ctx_stack[-1]["except"][0]:
                            ctx_stack[-1]["end"] = op_idx + arg + 2
                        else:
                            ctx_stack[-1]["else"] = op_idx + arg
                    elif ctx_stack[-1]["kind"] == "ctrl":
                        ctx_stack[-1]["end"] = op_idx + arg + 2
                    if (
                        len(bytecode[offset:]) >= 6
                        and dis.opname[bytecode[offset]] == "POP_TOP"
                        and dis.opname[bytecode[offset + 2]] == "POP_TOP"
                        and dis.opname[bytecode[offset + 4]] == "POP_TOP"
                    ):
                        [*_] = read(6)
                    # next_instruction_idx = offset
                elif opname in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
                    idx, test_node = stack.pop()
                    if version >= (3, 10):
                        matching_ctrls = [
                            ctx
                            for ctx in ctx_stack
                            if ctx["kind"] == "ctrl"
                            and ctx["body_start"] == arg
                        ]
                        print("MATCHING CTRLS", matching_ctrls)
                        if len(matching_ctrls):
                            ctx = matching_ctrls[0]
                            inst = next(i for idx, i in instructions if idx == ctx["start"])
                            print("INST", "inst", inst, "test", test_node)
                            if (
                                  opname == "POP_JUMP_IF_TRUE" and ast.dump(inst) == ast.dump(test_node)
                                or opname == "POP_JUMP_IF_FALSE" and ast.dump(inst) == ast.dump(test_node)
                            ):
                                process_jump_absolute(opname, ctx["start"])
                                break

                    if opname == "POP_JUMP_IF_TRUE":
                        test_node = ast.UnaryOp(op=ast.Not(), operand=test_node)
                    if (
                        isinstance(test_node, ast.Compare)
                        and test_node.ops[0] == "exception match"
                        and ctx_stack[-1]["kind"] == "try"
                    ):
                        exception = test_node.left
                        next_op = bytecode[offset + 2]
                        if next_op == dis.opmap["STORE_FAST"]:
                            [_1, _2, _3, var_idx] = read(4)
                        else:
                            var_idx = None

                        # IF NEXT THREE OPS ARE POP_TOP, skip them
                        next_ops = [
                            dis.opname[bytecode[offset + i]] for i in range(0, 6, 2)
                        ]
                        if next_ops == ["POP_TOP"] * 3:
                            offset += 6

                        ctx_stack[-1]["except_types"][len(ctx_stack[-1]["except"]) - 1] = (
                            exception,
                            code.co_varnames[var_idx] if var_idx is not None else None,
                        )
                        if (
                            dis.opname[bytecode[arg]] != "END_FINALLY"
                            and dis.opname[bytecode[arg - 2]] != "BEGIN_FINALLY"
                            and dis.opname[bytecode[arg]] != "POP_FINALLY"
                        ):
                            ctx_stack[-1]["except"].append(arg)
                    else:
                        start_else = None

                        # We're in a condition:
                        if arg > op_idx:
                            # - if we jump forward, the target is the end / start of else
                            start_else = arg

                        # Try to find the end if we're at the top of a loop
                        if start_else is None and idx in loop_ends:
                            start_else = loop_ends.get(idx)

                        # Try to find the end if we're moving to the top of a loop
                        if start_else is None:
                            start_else = loop_ends.get(arg)

                        assert start_else is not None
                        # TODO: check if we should really pop
                        add_instruction(test_node, start=idx)
                        ctx_stack.append(
                            {
                                "kind": "ctrl",
                                "body_start": op_idx + 2,
                                "start": idx,
                                "else": start_else,
                                "is_for": False,
                                "loop": False,
                                "has_test": True,
                            }
                        )

                        if arg > op_idx and idx in loop_ends and arg > loop_ends[idx]:
                            # If we're in a rare case in which we incorrectly predicted
                            # the end of a loop before running the decompilation, and find
                            # out now that the loop ends further down, we need to update
                            # the loop_ends dict
                            loop_ends[idx] = arg
                            ctx_stack.append(
                                {
                                    "kind": "ctrl",
                                    "body_start": op_idx + 2,
                                    "start": idx,
                                    "else": arg,
                                    "is_for": False,
                                    "loop": True,
                                    "has_test": False,
                                }
                            )

                        if arg == idx:
                            if (
                                # we're jumping back to the start of a loop
                                arg in loop_ends
                                # and we're not at the end of this loop
                                and idx != loop_ends[arg]
                                # and no loop exist in the stack with equal boundaries
                                and not any(
                                    ctx["start"] == arg and ctx["else"] == loop_ends[arg]
                                    and ctx["loop"]
                                    for ctx in ctx_stack
                                )
                            ):
                                print("CREATING A NEW LOOP !")
                                ctx_stack.append(
                                    {
                                        "kind": "ctrl",
                                        "body_start": op_idx + 2,
                                        "start": arg,
                                        "else": loop_ends[arg],
                                        "is_for": False,
                                        "loop": True,
                                        "has_test": False,
                                    }
                                )
                                #start_else = loop_ends[arg]
                elif opname == "JUMP_ABSOLUTE":
                    """
                    To decide if this is a break or a continue:
                    - we jump to the end of an the current loop => break
                    - we jump to the start of the upper loop => break
                    - we jump to the end
                    """
                    process_jump_absolute(opname, arg)
                elif opname == "COMPARE_OP":
                    if compareop_to_ast[arg] != "exception match":
                        right = stack.pop()[1]
                        left_idx, left = stack.pop()
                    else:
                        left_idx, left = stack.pop()
                        # right = stack.pop()[1]
                        right = None
                    add_stack(
                        ast.Compare(
                            left=left,
                            ops=[compareop_to_ast[arg]],
                            comparators=[right],
                        ),
                        start=left_idx,
                    )
                elif opname == "END_FINALLY":
                    if (
                        len(ctx_stack)
                        and ctx_stack[-1]["kind"] == "try"
                        and "end" not in ctx_stack[-1]
                    ):
                        ctx_stack[-1]["end"] = op_idx
                    # End of block => reset instruction
                    # next_instruction_idx = offset + 1
                elif opname == "CALL_FINALLY":
                    ctx_stack[-1]["begin_finally"] = op_idx + arg + 2
                    if bytecode[offset] == dis.opmap["POP_TOP"]:
                        [*_] = read(2)
                elif opname == "DUP_TOP":
                    if len(stack):
                        stack.append(stack[-1])
                elif opname == "BEGIN_FINALLY":
                    # next_instruction_idx = offset + 1
                    ctx_stack[-1]["begin_finally"] = op_idx
                    # end will be set by END_FINALLY
                    if "end" in ctx_stack[-1]:
                        del ctx_stack[-1]["end"]
                elif opname == "GET_ITER":
                    # During eval, pop and applies iter() to TOS
                    pass
                elif opname == "FOR_ITER":
                    start = op_idx
                    add_instruction(stack[-1][1], start=start)
                    stack.pop()
                    # Add one fake items to the stack for the target var
                    add_stack(None, start + 2)

                    start_else = op_idx + arg + 2
                    ctx_stack.append(
                        {
                            "kind": "ctrl",
                            "body_start": op_idx + 2,
                            "start": start,
                            "else": start_else,
                            "is_for": True,
                            "loop": True,
                            "has_test": False,
                        }
                    )
                elif opname == "MAKE_FUNCTION":
                    assert arg in (0, 8), arg
                    stack.pop()  # function name
                    func_code: ast.Constant = stack.pop()[1]
                    if arg == 8:
                        stack.pop()
                    assert isinstance(func_code, ast.Constant)
                    [sub_function_ast] = build_ast(func_code.value, as_function=True)
                    print("--------")
                    print(astunparse.unparse(sub_function_ast))
                    print("--------")
                    add_stack(sub_function_ast, op_idx)
                elif opname == "FORMAT_VALUE":
                    fmt_spec = None
                    conversion = -1
                    if arg & 0x03 == 0x00:
                        conversion = -1
                    elif arg & 0x03 == 0x01:
                        conversion = 115
                    elif arg & 0x03 == 0x02:
                        conversion = 114
                    elif arg & 0x03 == 0x03:
                        conversion = 97
                    if arg & 0x04 == 0x04:
                        fmt_spec = stack.pop()[1]
                        if not isinstance(fmt_spec, ast.JoinedStr):
                            fmt_spec = ast.JoinedStr([fmt_spec])
                    idx, value = stack.pop()
                    add_stack(
                        ast.JoinedStr(
                            [
                                ast.FormattedValue(
                                    value=value,
                                    conversion=conversion,
                                    format_spec=fmt_spec,
                                )
                            ]
                        ),
                        start=idx,
                    )
                elif opname == "BUILD_STRING":
                    indices, values = unzip_stack(arg)
                    values = [
                        part
                        for v in values
                        for part in (v.values if isinstance(v, ast.JoinedStr) else [v])
                    ]
                    add_stack(
                        ast.JoinedStr(values=values),
                        start=indices[0] if indices else op_idx,
                    )
                elif opname == "BUILD_SLICE":
                    indices, values = unzip_stack(arg)
                    values = list(values)
                    if arg == 2:
                        values = values + [None]
                    add_stack(ast.Slice(*values), start=indices[0] if indices else op_idx)
                elif opname == "ROT_TWO":
                    if dis.opname[bytecode[offset]] in (
                        "POP_EXCEPT",
                        "POP_BLOCK",
                        "POP_TOP",
                    ):
                        # skip this instruction
                        [*_] = read(2)
                    else:
                        s = stack
                        s[-1], s[-2] = s[-2], s[-1]
                elif opname == "ROT_THREE":
                    if dis.opname[bytecode[offset]] in (
                        "POP_EXCEPT",
                        "POP_BLOCK",
                        "POP_TOP",
                    ):
                        # skip this instruction
                        [*_] = read(2)
                    else:
                        s = stack
                        s[-1], s[-2], s[-3] = s[-2], s[-3], s[-1]
                elif opname == "ROT_FOUR":
                    if dis.opname[bytecode[offset]] in (
                        "POP_EXCEPT",
                        "POP_BLOCK",
                        "POP_TOP",
                    ):
                        # skip this instruction
                        [*_] = read(2)

                    else:
                        s = stack
                        s[-1], s[-2], s[-3], s[-4] = s[-2], s[-3], s[-4], s[-1]
                elif opname == "LOAD_CLOSURE":
                    add_stack(None, op_idx)
                elif opname == "RAISE_VARARGS":
                    exc = cause = None
                    assert 0 <= arg <= 2
                    if arg == 0:
                        idx = op_idx
                    elif arg == 1:
                        idx, exc = stack.pop()
                    elif arg == 2:
                        idx, cause = stack.pop()[1]
                        exc = stack.pop()[1]
                    add_instruction(ast.Raise(exc=exc, cause=cause), op_idx)
                elif opname == "SETUP_LOOP":
                    # Python 3.7
                    pass
                elif opname == "BREAK_LOOP":
                    # Python 3.7
                    add_instruction(ast.Break(), op_idx)
                else:
                    raise UnknownException()
                break

            if sorted(instructions, key=lambda x: x[0]) != instructions:
                print("SORTING:")
                print("BEFORE:", instructions)
                instructions = sorted(instructions, key=lambda x: x[0])
                print("AFTER :", instructions)

            last_inst_idx = instructions[-1][0] if len(instructions) else None
            if (
                  last_inst_idx in loop_ends and
                    op_idx < loop_ends[last_inst_idx] and
                  not any(ctx["start"] == last_inst_idx and ctx.get("else", ctx.get("end")) == loop_ends[last_inst_idx]
                          for ctx in ctx_stack)
            ):
                print("MISSING LOOP", last_inst_idx, loop_ends[last_inst_idx])
                ctx_stack.append(
                    {
                        "kind": "ctrl",
                        "body_start": op_idx + 2,
                        "start": last_inst_idx,
                        "else": loop_ends[last_inst_idx],
                        "is_for": False,
                        "loop": True,
                        "has_test": False,
                    }
                )



    except:
        raise UnknownException()



    if as_function:
        func_def = ast.FunctionDef(
            name=code.co_name,
            args=ast.arguments(
                args=[
                    ast.arg(arg=arg, annotation=None)
                    for arg in code.co_varnames[: code.co_argcount]
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[node for i, node in instructions],
            decorator_list=[],
            returns=None,
        )
        instructions = [(0, func_def)]  # type: ignore

    return [node for i, node in instructions]


def decompile(code: CodeType, as_function: bool = False) -> str:
    instructions = build_ast(code, as_function=as_function)
    print("---------")
    for inst in instructions:
        print(ast.dump(inst))
    try:
        return astunparse.unparse(instructions)
    except:
        raise Exception("\n".join(ast.dump(i) for i in instructions))


import ast
import inspect
import textwrap


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


def get_factory_code(code):
    function_body = decompile(code)

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
            f"def _factory_():\n"
            + (f"  {free_vars}\n" if free_vars else "")
            + f"{textwrap.indent(function_code, '  ')}\n"
            f"  return {function_name}\n"
            f"_fn_ = _factory_()\n"
        )
    else:
        factory_code = f"{function_code}\n" f"_fn_ = {function_name}\n"

    return factory_code
