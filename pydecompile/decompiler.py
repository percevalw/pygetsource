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
    print("----------------------")
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

    def add_instruction(inst, start=None):
        nonlocal next_instruction_idx
        instructions.append((next_instruction_idx if start is None else start, inst))
        next_instruction_idx = offset

    def add_stack(node, start=None):
        stack.append((op_idx if start is None else None, node))

    ctx_stack = []
    instructions: List[Tuple[int, ast.AST]] = []
    stack: List[Tuple[int, ast.AST]] = []
    op_idx = 0
    next_instruction_idx = 0
    try:
        for op, arg in read(batched=2):
            opname = dis.opname[op]
            op_idx = offset - 2

            if opname in ("LIST_APPEND", "SET_ADD") and ctx_stack[-1]["kind"] == "condition":
                # assume we're in a for-loop
                # We're in a comprehension, and don't do anything with the list_idx
                # index at this point. We only fake a yield to retrieve it later
                # when the comprehension is called from the parent function.
                new_stack = []
                for node_idx, node in stack:
                    if node_idx >= ctx_stack[-1]["start"]:
                        add_instruction(ast.Expr(ast.Yield(value=node)), node_idx)
                    else:
                        new_stack.append((node_idx, node))
                instructions = sorted(instructions, key=lambda x: x[0])
                #add_instruction(ast.Expr(ast.Yield(value=el_stack.pop()[1])))
                stack = new_stack

                # Update op to skip processing the LIST_APPEND and SET_ADD
                [op, arg] = read(2)
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

            # ==================== #

            for ctx in ctx_stack[::-1]:
                if ctx["kind"] == "try":
                    #print("START EXCEPTS", op_idx, start_excepts, ctx)
                    if op_idx in ctx["start_except"]:
                        ctx["already_popped"] = False
                        add_stack(EXC)
                        add_stack(EXC)
                        add_stack(EXC)
                        add_stack(EXC)
                        add_stack(EXC)
                        add_stack(EXC)

            new_ctx_stack = []
            for ctx in ctx_stack[::-1]:
                if ctx["kind"] == "try" and "end" in ctx and ctx.get("end") <= op_idx:
                    print("^^^ TRY CTX", ctx)
                    print("INSTRUCTIONS", [(i, (ast.dump(x) if x is not None else x)) for i, x in instructions])
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
                        print("ELSE/EXCEPT", [ast.dump(x) if isinstance(x, ast.AST) else None for x in else_block], [ast.dump(x) if isinstance(x, ast.AST) else None for e in except_blocks for x in e])
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
                        ctx["start"],
                    )
                elif (
                      ctx["kind"] == "condition"
                      and (
                            "end" in ctx and ctx["end"] <= op_idx or
                            "end" not in ctx and "start_else" in ctx and ctx["start_else"] == op_idx
                      )
                ):
                    print("^^^ COND_CTX", "at", op_idx, "ctx:", ctx)
                    print("INSTRUCTIONS", instructions)

                    if ctx["is_for"]:
                        iterator = next(
                            inst for idx, inst in instructions if ctx["start"] == idx
                        )
                        target_idx, target = next(
                            (idx, inst)
                            for idx, inst in instructions
                            if ctx["start"] < idx
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
                            ctx["start"],
                        )
                    else:
                        condition = next(
                            inst for idx, inst in instructions if idx == ctx["start"]
                        )
                        body_block = [
                            inst
                            for idx, inst in instructions
                            if ctx["start"] + 1 <= idx < ctx["start_else"]
                        ] or [ast.Pass()]
                        if "end" in ctx:
                            else_block = [
                                inst
                                for idx, inst in instructions
                                if ctx["start_else"] <= idx <= ctx["end"]
                            ]
                        else:
                            else_block = []
                        instructions = [i for i in instructions if i[0] < ctx["start"]]
                        if ctx.get("loop"):
                            add_instruction(
                                postprocess_loop(ast.While(
                                    test=condition,
                                    body=body_block,
                                    orelse=else_block or None,
                                )),
                                ctx["start"],
                            )
                        else:
                            add_instruction(
                                ast.If(
                                    test=condition,
                                    body=body_block,
                                    orelse=else_block or None,
                                ),
                                ctx["start"],
                            )
                else:
                    new_ctx_stack.append(ctx)

            ctx_stack = new_ctx_stack[::-1]

            ##################################

            # LOADING
            if opname == "LOAD_FAST":
                add_stack(ast.Name(code.co_varnames[arg]))
            elif opname == "LOAD_NAME":
                add_stack(ast.Name(code.co_names[arg]))
            elif opname == "LOAD_DEREF":
                add_stack(ast.Name(code.co_freevars[arg]))

            # ASSIGNMENTS
            elif opname == "STORE_FAST":
                if isinstance(stack[-1][1], ast.AugAssign):
                    add_instruction(stack.pop()[1])
                else:
                    targets = [ast.Name(code.co_varnames[arg])]
                    values = [stack.pop()[1]]
                    while dis.opname[bytecode[offset]] == "STORE_FAST":
                        op, arg = read(2)
                        targets.append(ast.Name(code.co_varnames[arg]))
                        values.append(stack.pop()[1])
                    add_instruction(ast.Assign(
                        targets=[ast.Tuple(targets) if len(targets) > 1 else targets[0]],
                        value=ast.Tuple(values) if len(targets) > 1 else values[0],
                    ))
            elif opname == "STORE_ATTR":
                attr = ast.Attribute(value=stack.pop()[1], attr=code.co_names[arg])
                add_instruction(ast.Assign(targets=[attr], value=stack.pop()[1]))
            elif opname == "STORE_SUBSCR":
                attr = ast.Subscript(slice=stack.pop()[1], value=stack.pop()[1])
                add_instruction(ast.Assign(targets=[attr], value=stack.pop()[1]))

            # DELETIONS
            elif opname == "DELETE_FAST":
                add_instruction(ast.Delete([ast.Name(code.co_varnames[arg])]))
            elif opname == "DELETE_ATTR":
                attr = ast.Attribute(value=stack.pop()[1], attr=code.co_names[arg])
                add_instruction(ast.Delete([attr]))
            elif opname == "DELETE_SUBSCR":
                attr = ast.Subscript(slice=stack.pop()[1], value=stack.pop()[1])
                add_instruction(ast.Delete([attr]))

            elif opname == "LOAD_GLOBAL":
                add_stack(ast.Name(code.co_names[arg]))
            elif opname == "LOAD_CONST":
                add_stack(ast.Constant(code.co_consts[arg], kind=None))

            # ATTRIBUTES
            elif opname == "LOAD_ATTR":
                attr = ast.Attribute(value=stack.pop()[1], attr=code.co_names[arg])
                add_stack(attr)

            elif opname == "LOAD_METHOD":
                add_stack(ast.Attribute(value=stack.pop()[1], attr=code.co_names[arg]))
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
                        )
                    )  # do we pop ?
                else:
                    raise UnknownException()
            elif opname == "UNPACK_EX":
                # FROM Python's doc:
                # The low byte of counts is the number of values before the list value,
                # the high byte of counts the number of values after it. The resulting
                # values are put onto the stack right-to-left.
                before, after = arg & 0xFF, arg >> 8
                print("BEFORE/AFTER", before, after)
                next_ops = list(read((before + after + 1) * 2))
                assert all(x == dis.opmap["STORE_FAST"] for x in next_ops[0::2])
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
                        value=stack.pop()[1],
                    )
                )



            # BUILD COLLECTIONS
            elif opname == "BUILD_TUPLE":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                add_stack(ast.Tuple(args))
            elif opname == "BUILD_LIST":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                add_stack(ast.List(args))
            elif opname == "BUILD_SET":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                add_stack(ast.Set(args) if len(args) >= 1 else ast.Call(func=ast.Name("set"), args=args, keywords=[]))
            elif opname == "BUILD_MAP":
                args = [stack.pop()[1] for _ in range(arg*2)][::-1]
                add_stack(ast.Dict(args[::2], args[1::2]))
            elif opname == "BUILD_CONST_KEY_MAP":
                keys = [ast.Constant(key, kind=None) for key in stack.pop()[1].value]
                values = [stack.pop()[1] for _ in range(arg)][::-1]
                add_stack(ast.Dict(keys, values))
            elif opname in ("BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL"):
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                items = extract_items(args)
                add_stack(ast.Tuple(items))
            elif opname == "BUILD_LIST_UNPACK":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                items = extract_items(args)
                add_stack(ast.List(items))
            elif opname == "BUILD_SET_UNPACK":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                items = extract_items(args)
                add_stack(ast.Set(items))
            elif opname in ("BUILD_MAP_UNPACK", "BUILD_MAP_UNPACK_WITH_CALL"):
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                keys, values = zip(*(
                    (k, v)
                    for x in args
                    for k, v in (
                        zip(x.keys, x.values) if isinstance(x, ast.Dict) else
                        zip(x.value.keys(), x.value.values()) if isinstance(x, ast.Constant) else
                        ((None, x,),))
                ))
                add_stack(ast.Dict(keys, values))
            # INPLACE/BINARY OPERATIONS
            elif op in binop_to_ast:
                right = stack.pop()[1]
                left = stack.pop()[1]
                add_stack(ast.BinOp(left=left, op=binop_to_ast[op], right=right))
            elif op in inplace_to_ast:
                right = stack.pop()[1]
                left = stack.pop()[1]
                add_stack(
                    ast.AugAssign(target=left, op=inplace_to_ast[op], value=right)
                )
            elif opname == "BINARY_SUBSCR":
                slice = stack.pop()[1]
                value = stack.pop()[1]
                add_stack(
                    ast.Subscript(value, slice)
                )
            elif op in unaryop_to_ast:
                add_stack(ast.UnaryOp(op=unaryop_to_ast[op], operand=stack.pop()[1]))

            # FUNCTIONS
            elif opname == "RETURN_VALUE":
                add_instruction(ast.Return(value=stack.pop()[1]))
            elif opname == "CALL_FUNCTION" or opname == "CALL_METHOD":
                args = [stack.pop()[1] for _ in range(arg)][::-1]
                func = stack.pop()[1]
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
                            #print(ast.dump(elt))
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
                    add_stack(tree)
                else:
                    add_stack(
                        ast.Call(
                            func=func,
                            args=args,
                            keywords=[],
                        )
                    )
            elif opname == "CALL_FUNCTION_KW":
                keys = stack.pop()[1].value
                values = [stack.pop()[1] for _ in range(len(keys))][::-1]
                args = [stack.pop()[1] for _ in range(arg - len(keys))][::-1]
                func = stack.pop()[1]
                add_stack(
                    ast.Call(
                        func=func,
                        args=args,
                        keywords=[
                            ast.keyword(arg=key, value=value)
                            for key, value in zip(keys, values)
                        ],
                    )
                )
            elif opname == "CALL_FUNCTION_EX":
                if arg & 0x01:
                    kwargs = stack.pop()[1]
                    args = stack.pop()[1]
                else:
                    kwargs = None
                    args = stack.pop()[1]
                func = stack.pop()[1]
                if isinstance(kwargs, ast.Dict):
                    kwargs = [
                        ast.keyword(arg=key.value if key is not None else None, value=value)
                        for key, value in zip(kwargs.keys, kwargs.values)
                    ] if kwargs is not None else []
                else:
                    kwargs = [ast.keyword(arg=None, value=kwargs)]
                add_stack(
                    ast.Call(
                        func=func,
                        args=args.elts,
                        keywords=kwargs,
                    )
                )

            # Control structures
            elif opname == "SETUP_FINALLY":
                ctx_stack.append(
                    {
                        "start": next_instruction_idx,
                        "end_body": op_idx + arg + 2,
                        "end_except": [],
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
                # ctx_stack[-1]["end_body"] = next_instruction_idx
            elif opname == "POP_EXCEPT":
                # TODO we should pop extraneous elements from the stack ?
                #while len(stack) and stack[-1][1] is EXC:
                #    stack.pop()
                ctx = ctx_stack[-1]
                assert ctx["kind"] == "try"
                if not ctx["already_popped"]:
                    stack.pop()
                    stack.pop()
                    stack.pop()
                    ctx["already_popped"] = True
            elif opname == "POP_FINALLY":
                print("STACK", stack)
                if arg: # preserve_tos
                    tos = stack.pop()
                # TODO we should pop extraneous elements from the stack ?
                if stack and stack[-1][1] is EXC:
                    stack.pop()
                elif isinstance(stack[-1][1], ast.Name):
                    exc_names = (pair[0].id for pair in ctx["except_types"].values())
                    print("EXC NAMES", exc_names)
                    # We check if the exception is an exception
                    if stack[-1][1].id in exc_names:
                        for _ in range(6):
                            stack.pop()
                if arg:
                    stack.append(tos)
                #ctx_stack[-1]["end_except"].append(next_instruction_idx)
            # JUMPS
            elif opname == "JUMP_FORWARD":
                #print("JUMP_FORWARD:", op_idx, op_idx + delta, ctx_stack)
                # try/except without Exception type
                assert ctx_stack[-1]["kind"] in ("try", "condition")
                if ctx_stack[-1]["kind"] == "try":
                    if ctx_stack[-1]["start_except"] and op_idx > ctx_stack[-1]["start_except"][0]:
                        ctx_stack[-1]["end"] = op_idx + arg + 2
                    else:
                    #    print("JUMP_FORWARD", op_idx + delta + 4, dis.opname[bytecode[op_idx + delta + 4]])
                    #    #if bytecode[op_idx + delta + 4] != dis.opmap["BEGIN_FINALLY"]:
                    #        # Fake push Exception and compare_op result on stack
                    #    add_stack(None)
                    #    add_stack(None)
                    #    #add_stack(None)
                        ctx_stack[-1]["start_else"] = op_idx + arg
                elif ctx_stack[-1]["kind"] == "condition":
                    ctx_stack[-1]["end"] = op_idx + arg + 2
                next_instruction_idx = offset
            elif opname in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
                test_node = stack.pop()[1]
                if opname == "POP_JUMP_IF_TRUE":
                    test_node = ast.UnaryOp(op=ast.Not(), operand=test_node)
                #print("IS COMPARE", isinstance(test_node, ast.Compare))
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
                    start = next_instruction_idx
                    add_instruction(test_node)
                    ctx_stack.append(
                        {
                            "kind": "condition",
                            "start": start,
                            "start_else": start_else,
                            "is_for": False,
                        }
                )
            elif opname == "JUMP_ABSOLUTE":
                #print("JUMP_ABSOLUTE", op_idx, jump_to, ctx_stack)

                if len(ctx_stack) and ctx_stack[-1]["kind"] == "condition" and ctx_stack[-1]["start_else"] == arg:
                    # If we jump at the else block / after the body block, this is a
                    # break.
                    # TODO: should we check if this is a loop ?

                    add_instruction(ast.Break())
                elif any(ctx["start"] == arg for ctx in ctx_stack):
                    # If we jump back to the start of a block, this is
                    # an indication that the upper block is a loop.
                    ctx = None
                    for ctx in ctx_stack[::-1]:
                        if ctx["start"] == arg:
                            ctx["loop"] = True
                            # TEST IF WE'RE ABOUT TO END A LOOP
                            # THEN, WE DON'T NEED TO CONTINUE
                            if not (ctx["start_else"] == offset):
                                add_instruction(ast.Continue())
                            break
                # elif jump_to < op_idx:
                #     print("> C", jump_to, op_idx)
                #     # We're in a top-level loop
                #     for ctx in ctx_stack:
                #         if ctx["start"] == jump_to:
                #             ctx["loop"] = True
                else:
                    raise UnknownException()
                    #print("> D", jump_to, op_idx)
                    # We're in a nested loop
                    # is_break = any(
                    #     ctx["start_else"] == jump_to for ctx in ctx_stack[::-1]
                    # )
                    # if is_break:
                    #     add_instruction(ast.Break())
                    # else:
                    #     raise UnknownException()
                #print("======>", op_idx, jump_to, ctx_stack)
            elif opname == "COMPARE_OP":
                if compareop_to_ast[arg] != "exception match":
                    right = stack.pop()[1]
                    left = stack.pop()[1]
                else:
                    left = stack.pop()[1]
                    right = stack.pop()[1]
                    #right = None
                add_stack(
                    ast.Compare(
                        left=left,
                        ops=[compareop_to_ast[arg]],
                        comparators=[right],
                    )
                )
            elif opname == "END_FINALLY":
                if len(ctx_stack) and ctx_stack[-1]["kind"] == "try" and "end" not in ctx_stack[-1]:
                    ctx_stack[-1]["end"] = next_instruction_idx
                # End of block => reset instruction
                next_instruction_idx = offset + 1
            elif opname == "CALL_FINALLY":
                ctx_stack[-1]["begin_finally"] = op_idx + arg + 2
            elif opname == "DUP_TOP":
                add_stack(stack[-1][1])
            elif opname == "BEGIN_FINALLY":
                next_instruction_idx = offset + 1
                ctx_stack[-1]["begin_finally"] = op_idx
                # end will be set by END_FINALLY
                if "end" in ctx_stack[-1]:
                    del ctx_stack[-1]["end"]
                add_stack(EXC)

            elif opname == "POP_TOP":
                # TODO should we check if we're in a loop ?
                if offset < len(bytecode) and dis.opname[bytecode[offset]] == "JUMP_ABSOLUTE":
                    [_, jump_to] = read(2)
                    if any(ctx["start"] == jump_to for ctx in ctx_stack):
                        # This is a break
                        print("THIS BREAK BECAUSE GO BACK TO TEST OF UPPER LOOP")
                        add_instruction(ast.Break())
                    # elif ctx_stack[-1]["start"] == jump_to:
                    #     # This is a continue
                    #     print("THIS CONTINUE BECAUSE GO BACK TO START OF CURRENT LOOP")
                    #     add_instruction(ast.Continue())
                    elif any(ctx["start_else"] == jump_to for ctx in ctx_stack):
                        # This is a break
                        print("THIS BREAK BECAUSE GO BACK TO END OF CURRENT LOOP")
                        add_instruction(ast.Break())
                    else:
                        raise UnknownException()
                # is this the correct way to handle this?
                else:
                    # Maybe this is one of those cases where pop_top occurs at the end
                    # after the return and is therefore never called ?
                    if len(stack) > 0:
                        top = stack.pop()[1]
                        if top is not None and top is not EXC and not isinstance(top, ast.Constant):
                            add_instruction(ast.Expr(top))

            elif opname.startswith("<"):
                pass
            # FOR LOOP
            elif opname == "GET_ITER":
                # During eval, pop and applies iter() to TOS
                pass
                #stack.pop()
                #add_stack(None)
            elif opname == "FOR_ITER":
                next_instruction_idx = op_idx
                start = next_instruction_idx
                #[_, next_op] = read(2)
                #print("FOR_ITER", op_idx, ctx_stack)
                #if next_op == dis.opmap["FOR_ITER"]:
                add_instruction(stack[-1][1])
                add_stack(None)
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
                #print("=>", next_instruction_idx)
            elif opname == "MAKE_FUNCTION":
                if arg == 0:
                    func_code: ast.Constant = stack[-2][1]
                    assert isinstance(func_code, ast.Constant)
                    [sub_function_ast] = build_ast(func_code.value, as_function=True)
                    add_stack(sub_function_ast)
                    try:
                        print("UNPARSE SUB", astunparse.unparse([sub_function_ast]))
                    except:
                        raise Exception(ast.dump(sub_function_ast))
                else:
                    raise UnknownException()
            elif opname in "MAP_ADD" and ctx_stack[-1]["kind"] == "condition":
                #print("CTX", ctx_stack[-1])
                # assume we're in a for-loop
                # We're in a comprehension, and don't do anything with the list_idx
                # index at this point. We only fake a yield to retrieve it later
                # when the comprehension is called from the parent function.
                stack, pairs = split_stack(stack, ctx_stack[-1]["start"])
                for (key_idx, key), (value_idx, value) in zip(pairs[-2::-2], pairs[-1::-2]):
                    add_instruction(ast.Expr(ast.Yield(value=ast.Tuple([key, value]))), key_idx)
                #for (key_idx, key), (value_idx, value) in zip(stack[-2::-2], stack[::-2]):
                #    if key_idx >= ctx_stack[-1]["start"]:
                #        add_instruction(ast.Expr(ast.Yield(value=ast.Tuple([key, value]))), key_idx)
                #        print("INST", ast.dump(instructions[-1][1]))
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
                add_stack(ast.JoinedStr([ast.FormattedValue(
                    value=stack.pop()[1],
                    conversion=conversion,
                    format_spec=fmt_spec,
                )]))
            elif opname == "BUILD_STRING":
                values = [stack.pop()[1] for _ in range(arg)][::-1]
                values = [
                    part
                    for v in values
                    for part in (v.values if isinstance(v, ast.JoinedStr) else [v])
                ]
                add_stack(ast.JoinedStr(values=values))
            elif opname == "BUILD_SLICE":
                values = [stack.pop()[1] for _ in range(arg)][::-1]
                add_stack(ast.Slice(*values))
            elif opname == "ROT_TWO":
                s = stack
                s[-1], s[-2] = s[-2], s[-1]
            elif opname == "ROT_THREE":
                s = stack
                s[-1], s[-2], s[-3] = s[-2], s[-3], s[-1]
            elif opname == "ROT_FOUR":
                s = stack
                s[-1], s[-2], s[-3], s[-4] = s[-2], s[-3], s[-4], s[-1]
            else:
                raise UnknownException()


            print("INST", opname.ljust(20), op_idx, "STACK", [ast.dump(e)[:50] if isinstance(e, ast.AST) else e for i, e in stack], ctx_stack[-1].get("start_except") if ctx_stack else None)
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

    #print("FINAL STACK", [(i, ast.dump(el) if isinstance(el, ast.AST) else el) for i, el in stack])
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