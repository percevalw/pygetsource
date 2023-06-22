import ast
import dis
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
    cmp_map["in"]: ast.In(),
    cmp_map["not in"]: ast.NotIn(),
    cmp_map["is"]: ast.Is(),
    cmp_map["is not"]: ast.IsNot(),
    cmp_map["exception match"]: "exception match",
}

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
                if (
                    len(handler.body) == 1
                    and isinstance(handler.body[0], ast.Try)
                ):
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
                          isinstance(fb0, ast.Assign) and fb0.targets[0].id == handler.name
                          and fb0.value.value is None
                        and isinstance(fb1, ast.Delete) and fb1.targets[0].id == handler.name
                    ):
                        # If found, replace the body of the handler with the body of
                        # the child
                        handler.body = child.body or [ast.Pass()]
                        if (
                              isinstance(fb0, ast.Assign) and fb0.targets[0].id == handler.name
                              and fb0.value.value is None
                            and isinstance(fb1, ast.Delete) and fb1.targets[0].id == handler.name
                        ):
                            handler.body = child.body or [ast.Pass()]

        finally:
            pass
    return node

class ExceptionStub:
    def __repr__(self):
        return "EXC"

EXC = ExceptionStub()

def postprocess_loop(node):
    assert isinstance(node, (ast.While, ast.For))
    # Remove continue at the end of the loop
    #node.body = node.body[:-1]
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
                    if_node.body = if_node.body[:-1]
                    if_node.orelse = node.body[i+1:] or None
                    node.body = node.body[:i] + [if_node]
    return node

def extract_yield(node):
    if isinstance(node, list):
        assert len(node) == 1, f"Expected single statement instead of {ast.dump(node)}"
        node = node[0]
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


def build_ast(code: CodeType, as_function=True):
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
            f"Unsupported {opname} ({op}) at {op_idx}:\n{bytecode_str}\nCurrent ctx: {ctx_stack}\nCurrent stack: {stack}"
        )

    def add_instruction(inst, start: int):
        #nonlocal next_instruction_idx
        instructions.append((start, inst))
        print(" > INST", start, ast.dump(inst) if isinstance(inst, ast.AST) else None)
        #op_idx = offset

    def add_stack(node, start):
        stack.append((op_idx if start is None else start, node))

    def unzip_stack(n):
        elements = [stack.pop() for _ in range(n)][::-1]
        return zip(*elements) if len(elements) else ((), ())

    ctx_stack = []
    instructions: List[Tuple[int, ast.AST]] = []
    stack: List[Tuple[int, ast.AST]] = []
    op_idx = 0
    op_idx = 0
    try:
        for op, arg in read(batched=2):
            opname = dis.opname[op]
            op_idx = offset - 2

            print("CODE", opname.ljust(20), op_idx, "STACK", [(i, ast.dump(e)[:50]) if isinstance(e, ast.AST) else e for i, e in stack], ctx_stack)
#            if opname in ("LIST_APPEND", "SET_ADD") and ctx_stack[-1]["kind"] == "condition":
#                # assume we're in a for-loop
#                # We're in a comprehension, and don't do anything with the list_idx
#                # index at this point. We only fake a yield to retrieve it later
#                # when the comprehension is called from the parent function.
#                #new_stack = []
#                #for node_idx, node in stack:
#                #    if node_idx >= ctx_stack[-1]["start"]:
#                #        add_instruction(ast.Expr(ast.Yield(value=node)), node_idx)
#                #    else:
#                #        new_stack.append((node_idx, node))
#                #instructions = sorted(instructions, key=lambda x: x[0])
#                ##add_instruction(ast.Expr(ast.Yield(value=el_stack.pop()[1])))
#                #stack = new_stack
##
#                ## Update op to skip processing the LIST_APPEND and SET_ADD
#                #[op, arg] = read(2)
#                #opname = dis.opname[op]
#                #op_idx = offset - 2

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

            # ==================== #

            for ctx in ctx_stack[::-1]:
                if ctx["kind"] == "try":
                    if op_idx in ctx["start_except"]:
                        ctx["already_popped"] = False
                        # add_stack(EXC)
                        # add_stack(EXC)
                        # add_stack(EXC)
                        # add_stack(EXC)
                        # add_stack(EXC)
                        # add_stack(EXC)

            new_ctx_stack = []
            for ctx in ctx_stack[::-1]:
                if ctx["kind"] == "try" and "end" in ctx and ctx.get("end") <= op_idx:
                    body_block = [
                        inst
                        for idx, inst in instructions
                        if ctx["start"] <= idx < ctx["end_body"]
                    ]
                    if "start_else" in ctx:
                        else_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["start_else"] <= idx < ctx["end"]
                        ]
                    else:
                        else_block = []
                    if len(ctx["start_except"]):
                        except_blocks = [
                            [inst for idx, inst in instructions if start <= idx < end]
                            for start, end in zip(
                                ctx["start_except"],
                                ctx["start_except"][1:] + [ctx.get("start_else", ctx.get("end"))],
                            )
                        ]
                    else:
                        except_blocks = []
                    if "begin_finally" in ctx:
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
                        postprocess_try_finally(ast.Try(
                            body=body_block,
                            handlers=[
                                ast.ExceptHandler(
                                    type=ctx["except_types"].get(i)[0] if i in ctx["except_types"] else None,
                                    name=ctx["except_types"].get(i)[1] if i in ctx["except_types"] else None,
                                    body=except_block or [ast.Pass()],
                                )
                                for i, except_block in enumerate(except_blocks)
                            ],
                            orelse=else_block or None,
                            finalbody=finally_block,
                        )),
                        start=ctx["start"],
                    )
                elif (
                      ctx["kind"] == "condition"
                      and (
                            "end" in ctx and ctx["end"] <= op_idx or
                            "end" not in ctx and "start_else" in ctx and ctx["start_else"] == op_idx
                      )
                ):
                    print("DOING CONDITION", op_idx, ctx)

                    if ctx["is_for"]:
                        iterator = next(
                            inst for idx, inst in instructions if ctx["start"] == idx
                        )
                        target_idx, target = next(
                            (idx, inst)
                            for idx, inst in instructions
                            if ctx["start"] + 2 == idx
                        )
                        body_block = [
                            inst
                            for idx, inst in instructions
                            if target_idx < idx < ctx["start_else"]
                        ] or [ast.Pass()]
                        else_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["start_else"] <= idx <= ctx["end"]
                        ]
                        if isinstance(target, ast.Assign):
                            target = target.targets[0]
                        instructions = [i for i in instructions if i[0] < ctx["start"]]
                        add_instruction(
                            postprocess_loop(ast.For(
                                iter=iterator,
                                target=target,
                                body=body_block,
                                orelse=else_block or None,
                            )),
                            start=ctx["start"],
                        )
                    else:
                        condition = next(
                            inst for idx, inst in instructions if idx == ctx["start"]
                        )
                        is_ternary = False
                        body_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["start"] + 1 <= idx < ctx["start_else"]
                        ]
                        if not len(body_block):
                            stack_block = [
                                inst
                                for idx, inst in stack
                                if ctx["start"] + 1 <= idx < ctx["start_else"]
                            ]
                            assert len(stack_block) == 1
                            is_ternary = True
                            body_block = stack_block
                        if "end" in ctx:
                            else_block = [
                                inst
                                for idx, inst in instructions
                                if ctx["start_else"] <= idx <= ctx["end"]
                            ]

                            if not len(else_block):
                                else_stack_block = [
                                    inst
                                    for idx, inst in stack
                                    if ctx["start_else"] <= idx <= ctx["end"]
                                ]
                                if len(else_stack_block) == 1:
                                    is_ternary = True
                                    else_block = else_stack_block
                                else:
                                    else_block = []
                        else:
                            else_block = []

                        if is_ternary:
                            print("STACK", stack)
                            stack = [i for i in stack if i[0] < ctx["start"]]
                            print("STACK AFTER", stack)
                        instructions = [i for i in instructions if i[0] < ctx["start"]]

                        if ctx.get("loop"):
                            add_instruction(
                                postprocess_loop(ast.While(
                                    test=condition,
                                    body=body_block,
                                    orelse=else_block or None,
                                )),
                                start=ctx["start"],
                            )
                        elif not is_ternary:
                            print("AS STANDARD IF")
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
                    new_ctx_stack.append(ctx)

            ctx_stack = new_ctx_stack[::-1]

            ##################################

            # LOADING
            if opname == "LOAD_FAST":
                add_stack(ast.Name(code.co_varnames[arg]), start=op_idx)
            elif opname == "LOAD_NAME":
                add_stack(ast.Name(code.co_names[arg]), start=op_idx)
            elif opname == "LOAD_DEREF":
                add_stack(ast.Name(code.co_freevars[arg]), start=op_idx)

            # ASSIGNMENTS
            elif opname == "STORE_FAST":
                idx, value = stack.pop()
                if isinstance(value, ast.AugAssign):
                    add_instruction(value, start=idx)
                else:
                    targets = [ast.Name(code.co_varnames[arg])]
                    values = [value]
                    while dis.opname[bytecode[offset]] == "STORE_FAST":
                        op, arg = read(2)
                        targets.append(ast.Name(code.co_varnames[arg]))
                        idx, value = stack.pop()
                        values.append(value)
                    add_instruction(ast.Assign(
                        targets=[ast.Tuple(targets) if len(targets) > 1 else targets[0]],
                        value=ast.Tuple(values) if len(targets) > 1 else values[0],
                    ), start=idx)
            elif opname == "STORE_ATTR":
                idx, value = stack.pop()
                attr = ast.Attribute(value=value, attr=code.co_names[arg])
                add_instruction(ast.Assign(targets=[attr], value=stack.pop()[1]), start=idx)
            elif opname == "STORE_SUBSCR":
                idx, value = stack.pop()
                attr = ast.Subscript(slice=value, value=stack.pop()[1])
                add_instruction(ast.Assign(targets=[attr], value=stack.pop()[1]), start=idx)
            elif opname in ("LIST_APPEND", "SET_ADD") and ctx_stack[-1]["kind"] == "condition":
                idx, value = stack.pop()
                add_instruction(ast.Expr(ast.Yield(value)), start=idx)
            # DELETIONS
            elif opname == "DELETE_FAST":
                add_instruction(ast.Delete([ast.Name(code.co_varnames[arg])]), start=op_idx)
            elif opname == "DELETE_ATTR":
                idx, value = stack.pop()
                attr = ast.Attribute(value=value, attr=code.co_names[arg])
                add_instruction(ast.Delete([attr]), start=idx)
            elif opname == "DELETE_SUBSCR":
                idx, value = stack.pop()
                attr = ast.Subscript(slice=value, value=stack.pop()[1], start=idx)
                add_instruction(ast.Delete([attr]), start=idx)

            elif opname == "LOAD_GLOBAL":
                add_stack(ast.Name(code.co_names[arg]), start=op_idx)
            elif opname == "LOAD_CONST":
                add_stack(ast.Constant(code.co_consts[arg], kind=None), start=op_idx)

            # ATTRIBUTES
            elif opname == "LOAD_ATTR":
                idx, value = stack.pop()
                attr = ast.Attribute(value=value, attr=code.co_names[arg], start=idx)
                add_stack(attr, start=idx)

            elif opname == "LOAD_METHOD":
                idx, value = stack.pop()
                add_stack(ast.Attribute(value=value, attr=code.co_names[arg]), start=idx)
            elif opname == "UNPACK_SEQUENCE":
                next_ops = list(read(arg * 2))
                if all(x == dis.opmap["STORE_FAST"] for x in next_ops[0::2]):
                    add_instruction(
                        ast.Assign(
                            targets=[
                                ast.Tuple(
                                    [
                                        ast.Name(
                                            code.co_varnames[var_idx],
                                            ctx=ast.Load(),
                                        )
                                        for var_idx in next_ops[1::2]
                                    ]
                                )
                            ],
                            value=stack[-1][1],#stack.pop()[1],
                        ),
                        start=stack[-1][0],
                    )  # do we pop ?
                else:
                    raise UnknownException()
            elif opname == "UNPACK_EX":
                # FROM Python's doc:
                # The low byte of counts is the number of values before the list value,
                # the high byte of counts the number of values after it. The resulting
                # values are put onto the stack right-to-left.
                before, after = arg & 0xFF, arg >> 8
                next_ops = list(read((before + after + 1) * 2))
                assert all(x == dis.opmap["STORE_FAST"] for x in next_ops[0::2])
                idx, value = stack.pop()
                add_instruction(
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                [
                                    ast.Name(code.co_varnames[var_idx])
                                    if i != before
                                    else ast.Starred(ast.Name(code.co_varnames[var_idx]))
                                    for i, var_idx in enumerate(next_ops[1::2])
                                ]
                            )
                        ],
                        value=value,
                    ),
                    start=idx,
                )



            # BUILD COLLECTIONS
            elif opname == "BUILD_TUPLE":
                indices, args = unzip_stack(arg)
                print("INDICES", indices, "ARGS", args)
                add_stack(ast.Tuple(args), start=indices[0] if indices else op_idx)
            elif opname == "BUILD_LIST":
                indices, args = unzip_stack(arg)
                add_stack(ast.List(args), start=indices[0] if indices else op_idx)
            elif opname == "BUILD_SET":
                indices, args = unzip_stack(arg)
                add_stack(ast.Set(args) if len(args) >= 1 else
                          ast.Call(func=ast.Name("set"), args=args, keywords=[]),
                          start=indices[0] if indices else op_idx)
            elif opname == "BUILD_MAP":
                indices, args = unzip_stack(arg*2)
                add_stack(ast.Dict(args[::2], args[1::2]), start=indices[0] if indices else op_idx)
            elif opname == "BUILD_CONST_KEY_MAP":
                keys = [ast.Constant(key, kind=None) for key in stack.pop()[1].value]
                indices, values = unzip_stack(arg)
                add_stack(ast.Dict(keys, values), start=indices[0] if indices else op_idx)
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
                print("UNPACK DICT", args)
                keys, values = zip(*(
                    (k, v)
                    for x in args
                    for k, v in (
                        zip(x.keys, x.values) if isinstance(x, ast.Dict) else
                        zip(x.value.keys(), x.value.values()) if isinstance(x, ast.Constant) else
                        ((None, x,),))
                ))
                add_stack(ast.Dict(keys, values), start=indices[0] if indices else op_idx)
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

                        def visit_If(self, node: ast.If) -> ast.AST:
                            try:
                                value = extract_yield(node.body)
                                if node.orelse:
                                    has_else = True
                                    else_value = extract_yield(node.orelse)
                                else:
                                    has_else = False
                                    else_value = None
                                if has_else:
                                    value = ast.Expr(ast.Yield(ast.IfExp(
                                        test=node.test,
                                        body=value,
                                        orelse=else_value,
                                    )))
                                    return value
                                return node
                            except:
                                return super().generic_visit(node)

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
                            generator = ast.comprehension(
                                target=target,
                                iter=iter,
                                ifs=[condition] if condition is not None else [],  # TODO
                            )
                            #if not isinstance(elt, ast.IfExp):
                            #    raise Exception("Expected IfExp instead of " + ast.dump(elt))
                            # TODO handle DictComp
                            if isinstance(elt, cls):
                                elt.generators = [generator] + elt.generators
                            else:
                                elt = extract_yield(elt)
                            if cls is ast.DictComp:
                                assert isinstance(elt, ast.Tuple)
                                return cls(
                                    key=elt.elts[0],
                                    value=elt.elts[1],
                                    generators=[generator],
                                    ifs=[],
                                )
                            elif cls is ast.ListComp:
                                return cls(
                                    elt=elt,
                                    generators=[generator],
                                    ifs=[],
                                )
                            elif cls is ast.SetComp:
                                return cls(
                                    elt=elt,
                                    generators=[generator],
                                    ifs=[],
                                )
                            else:
                                raise Exception("Unknown comprehension type: {}".format(cls))

                    assert len(func.body) == 2
                    cls = (
                        ast.DictComp if isinstance(func.body[1].value, ast.Dict) else
                        ast.ListComp if isinstance(func.body[1].value, ast.List) else
                        ast.SetComp
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
                    ), start=idx,
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
                    kwargs = [
                        ast.keyword(arg=key.value if key is not None else None, value=value)
                        for key, value in zip(kwargs.keys, kwargs.values)
                    ] if kwargs is not None else []
                else:
                    kwargs = [ast.keyword(arg=None, value=kwargs)]
                add_stack(ast.Call(func=func, args=args.elts, keywords=kwargs,), start=idx)

            # Control structures
            elif opname == "SETUP_FINALLY":
                ctx_stack.append(
                    {
                        "start": op_idx,
                        "end_body": op_idx + arg + 2,
                        "start_except": [],
                        "except_types": {},
                        "kind": "try",
                        "already_popped": False,
                    }
                )
                if dis.opname[bytecode[op_idx + arg + 2]] != "END_FINALLY" and dis.opname[bytecode[op_idx + arg]] != "BEGIN_FINALLY" and dis.opname[bytecode[op_idx + arg + 2]] != "POP_FINALLY":
                    ctx_stack[-1]["start_except"].append(op_idx + arg + 2)
                #elif dis.opname[bytecode[op_idx + arg]] == "BEGIN_FINALLY" or dis.opname[bytecode[op_idx + arg + 2]] == "POP_FINALLY":
                #    ctx_stack[-1]["begin_finally"] = op_idx + arg
            elif opname == "POP_BLOCK":
                pass
            elif opname == "POP_TOP":
                # TODO should we check if we're in a loop ?
                if offset < len(bytecode) and dis.opname[bytecode[offset]] == "JUMP_ABSOLUTE":
                    [_, jump_to] = read(2)
                    if any(ctx["start"] == jump_to for ctx in ctx_stack):
                        # This is a break
                        add_instruction(ast.Break(), start=op_idx)
                    # elif ctx_stack[-1]["start"] == jump_to:
                    #     # This is a continue
                    #     add_instruction(ast.Continue())
                    elif any(ctx["start_else"] == jump_to for ctx in ctx_stack):
                        # This is a break
                        add_instruction(ast.Break(), start=op_idx)
                    else:
                        raise UnknownException()
                # is this the correct way to handle this?
                else:
                    # Maybe this is one of those cases where pop_top occurs at the end
                    # after the return and is therefore never called ?
                    if len(stack) > 0:
                        top = stack.pop()[1]
                        # if top is not None and top is not EXC and not isinstance(top, ast.Constant):
                        add_instruction(ast.Expr(top), start=op_idx)

                # ctx_stack[-1]["end_body"] = next_instruction_idx
            elif opname == "POP_EXCEPT":
                pass
            elif opname == "POP_FINALLY":
                pass
                #if arg: # preserve_tos
                #    tos = stack.pop()
                ## TODO we should pop extraneous elements from the stack ?
                ##if stack and stack[-1][1] is EXC:
                ##    stack.pop()
                #elif isinstance(stack[-1][1], ast.Name):
                #    exc_names = (pair[0].id for pair in ctx["except_types"].values())
                #    # We check if the exception is an exception
                #    if stack[-1][1].id in exc_names:
                #        for _ in range(6):
                #            stack.pop()
                #if arg:
                #    stack.append(tos)
            # JUMPS
            elif opname == "JUMP_FORWARD":
                # try/except without Exception type
                assert ctx_stack[-1]["kind"] in ("try", "condition")
                if ctx_stack[-1]["kind"] == "try":
                    if ctx_stack[-1]["start_except"] and op_idx > ctx_stack[-1]["start_except"][0]:
                        ctx_stack[-1]["end"] = op_idx + arg + 2
                    else:
                    #    #if bytecode[op_idx + delta + 4] != dis.opmap["BEGIN_FINALLY"]:
                    #        # Fake push Exception and compare_op result on stack
                    #    add_stack(None)
                    #    add_stack(None)
                    #    #add_stack(None)
                        ctx_stack[-1]["start_else"] = op_idx + arg
                elif ctx_stack[-1]["kind"] == "condition":
                    ctx_stack[-1]["end"] = op_idx + arg + 2
                if (
                      len(bytecode[offset:]) >= 6 and
                        dis.opname[bytecode[offset]] == "POP_TOP" and
                        dis.opname[bytecode[offset + 2]] == "POP_TOP" and
                        dis.opname[bytecode[offset + 4]] == "POP_TOP"):
                    [*_] = read(6)
                # next_instruction_idx = offset
            elif opname in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
                idx, test_node = stack.pop()
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

                    ctx_stack[-1]["except_types"][len(ctx_stack[-1]["start_except"])-1] = (
                        exception,
                        code.co_varnames[var_idx] if var_idx is not None else None,
                    )
                    if dis.opname[bytecode[arg]] != "END_FINALLY" and dis.opname[bytecode[arg - 2]] != "BEGIN_FINALLY" and dis.opname[bytecode[arg]] != "POP_FINALLY":
                        ctx_stack[-1]["start_except"].append(arg)
                else:
                    # We're in a condition:
                    if arg >= op_idx:
                        # - if we jump forward, the target is the end / start of else
                        start_else = arg
                    else:
                        # - if we jump backward, this is an if block without an else,
                        # inside a loop, and the end of the if block is the end of the
                        # loop
                        for ctx in ctx_stack[::-1]:
                            if ctx["start"] == arg and "start_else" in ctx:
                                start_else = ctx["start_else"]
                                break
                        else:
                            raise UnknownException()
                    # TODO: check if we should really pop
                    add_instruction(test_node, start=idx)
                    ctx_stack.append(
                        {
                            "kind": "condition",
                            "start": idx,
                            "start_else": start_else,
                            "is_for": False,
                        }
                )
            elif opname == "JUMP_ABSOLUTE":
                if len(ctx_stack) and ctx_stack[-1]["kind"] == "condition" and ctx_stack[-1]["start_else"] == arg:
                    # If we jump at the else block / after the body block, this is a
                    # break.
                    # TODO: should we check if this is a loop ?

                    add_instruction(ast.Break(), start=op_idx)
                elif any(ctx["start"] == arg for ctx in ctx_stack if ctx["kind"] == "condition"):
                    # If we jump back to the start of a block, this is
                    # an indication that the upper block is a loop.
                    ctx = None
                    for ctx in ctx_stack[::-1]:
                        if ctx["kind"] == "condition" and ctx["start"] == arg:
                            ctx["loop"] = True
                            # TEST IF WE'RE ABOUT TO END A LOOP
                            # THEN, WE DON'T NEED TO CONTINUE
                            if not ("start_else" in ctx and ctx["start_else"] == offset):
                                add_instruction(ast.Continue(), start=op_idx)
                            break
                # elif jump_to < op_idx:
                #     # We're in a top-level loop
                #     for ctx in ctx_stack:
                #         if ctx["start"] == jump_to:
                #             ctx["loop"] = True
                else:

                    # ctx_stack.append(
                    #     {
                    #         "kind": "condition",
                    #         "start": arg,
                    #         "start_else": None,
                    #         "is_for": False,
                    #         "loop": True,
                    #     })
                    # ctx_stack = sorted(ctx_stack, key=lambda ctx: ctx["start"])
                    # print("ERROR", [(ctx["start"], arg, ctx["kind"]) for ctx in ctx_stack])
                    raise UnknownException()
                    # We're in a nested loop
                    # is_break = any(
                    #     ctx["start_else"] == jump_to for ctx in ctx_stack[::-1]
                    # )
                    # if is_break:
                    #     add_instruction(ast.Break())
                    # else:
                    #     raise UnknownException()
            elif opname == "COMPARE_OP":
                if compareop_to_ast[arg] != "exception match":
                    right = stack.pop()[1]
                    left_idx, left = stack.pop()
                else:
                    left_idx, left = stack.pop()
                    #right = stack.pop()[1]
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
                if len(ctx_stack) and ctx_stack[-1]["kind"] == "try" and "end" not in ctx_stack[-1]:
                    ctx_stack[-1]["end"] = op_idx
                # End of block => reset instruction
                #next_instruction_idx = offset + 1
            elif opname == "CALL_FINALLY":
                ctx_stack[-1]["begin_finally"] = op_idx + arg + 2
                if bytecode[offset] == dis.opmap["POP_TOP"]:
                    [*_] = read(2)
            elif opname == "DUP_TOP":
                pass
            elif opname == "BEGIN_FINALLY":
                #next_instruction_idx = offset + 1
                ctx_stack[-1]["begin_finally"] = op_idx
                # end will be set by END_FINALLY
                if "end" in ctx_stack[-1]:
                    del ctx_stack[-1]["end"]
                #add_stack(EXC)

            elif opname.startswith("<"):
                pass
            # FOR LOOP
            elif opname == "GET_ITER":
                # During eval, pop and applies iter() to TOS
                pass
                #stack.pop()
                #add_stack(None)
            elif opname == "FOR_ITER":
                #next_instruction_idx = op_idx
                start = op_idx
                #[_, next_op] = read(2)
                #if next_op == dis.opmap["FOR_ITER"]:
                add_instruction(stack[-1][1], start=start)
                add_stack(None, start + 2)
#                names = []
#                if bytecode[offset] == dis.opmap["UNPACK_SEQUENCE"]:
#                    [_, arg] = read(1)
#                    for range(:
#                        [_, var_idx] = read(2)
#                        names.append(ast.Name(code.co_varnames[var_idx]))
#                else:
#                    [_, var_idx] = read(2)
#                    names.append(ast.Name(code.co_varnames[var_idx]))
#
                start_else = op_idx + arg + 2
#
#                add_instruction(ast.Assign(
#                    targets=names,
#                    value=stack[-1][1])
#                )
                ctx_stack.append(
                    {
                        "kind": "condition",
                        "start": start,
                        "start_else": start_else,
                        "is_for": True,
                    }
                )
            elif opname == "MAKE_FUNCTION":
                if arg == 0:
                    func_code: ast.Constant = stack[-2][1]
                    assert isinstance(func_code, ast.Constant)
                    [sub_function_ast] = build_ast(func_code.value, as_function=True)
                    add_stack(sub_function_ast, op_idx)
                    try:
                        print("SUB FUNCTION AST", astunparse.unparse(sub_function_ast))
                    except:
                        raise Exception(ast.dump(sub_function_ast))
                else:
                    raise UnknownException()
            elif opname in "MAP_ADD" and ctx_stack[-1]["kind"] == "condition":
                # assume we're in a for-loop
                # We're in a comprehension, and don't do anything with the list_idx
                # index at this point. We only fake a yield to retrieve it later
                # when the comprehension is called from the parent function.
                stack, pairs = split_stack(stack, ctx_stack[-1]["start"])
                for (key_idx, key), (value_idx, value) in zip(pairs[-2::-2], pairs[-1::-2]):
                    add_instruction(ast.Expr(ast.Yield(value=ast.Tuple([key, value]))), start=key_idx)
                #for (key_idx, key), (value_idx, value) in zip(stack[-2::-2], stack[::-2]):
                #    if key_idx >= ctx_stack[-1]["start"]:
                #        add_instruction(ast.Expr(ast.Yield(value=ast.Tuple([key, value]))), key_idx)
                #    else:
                #        new_stack.append((value_idx, value))
                #        new_stack.append((key_idx, key))
                instructions = sorted(instructions, key=lambda x: x[0])
                #add_instruction(ast.Expr(ast.Yield(value=el_stack.pop()[1])))
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
                add_stack(ast.JoinedStr([ast.FormattedValue(
                    value=value,
                    conversion=conversion,
                    format_spec=fmt_spec,
                )]), start=idx)
            elif opname == "BUILD_STRING":
                indices, values = unzip_stack(arg)
                values = [
                    part
                    for v in values
                    for part in (v.values if isinstance(v, ast.JoinedStr) else [v])
                ]
                add_stack(ast.JoinedStr(values=values), start=indices[0] if indices else op_idx)
            elif opname == "BUILD_SLICE":
                indices, values = unzip_stack(arg)
                add_stack(ast.Slice(*values), start=indices[0] if indices else op_idx)
            elif opname == "ROT_TWO":
                if dis.opname[bytecode[offset]] in ("POP_EXCEPT", "POP_BLOCK", "POP_TOP"):
                    # skip this instruction
                    [*_] = read(2)
                else:
                    s = stack
                    s[-1], s[-2] = s[-2], s[-1]
            elif opname == "ROT_THREE":
                if dis.opname[bytecode[offset]] in ("POP_EXCEPT", "POP_BLOCK", "POP_TOP"):
                    # skip this instruction
                    [*_] = read(2)
                else:
                    s = stack
                    s[-1], s[-2], s[-3] = s[-2], s[-3], s[-1]
            elif opname == "ROT_FOUR":

                if dis.opname[bytecode[offset]] in ("POP_EXCEPT", "POP_BLOCK", "POP_TOP"):
                    # skip this instruction
                    [*_] = read(2)

                else:
                    s = stack
                    s[-1], s[-2], s[-3], s[-4] = s[-2], s[-3], s[-4], s[-1]
            else:
                raise UnknownException()

            #print("CTX", ctx_stack)

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