import ast
import inspect


class Unpacking(ast.AST):
    _attributes = ()
    _fields = ("value",)

    def __init__(self, value, counter, starred=False):
        self.value = value
        self.counter = counter
        self.starred = starred


class ForTargetPlaceholder(ast.AST):
    _attributes = ()
    _fields = ()


class ExceptionMatch(ast.AST):
    _attributes = ()
    _fields = ("value",)

    def __init__(self, value):
        self.value = value


class ExceptionPlaceholder(ast.AST):
    _attributes = ()
    _fields = ()


class Reraise(ast.AST):
    _attributes = ()
    _fields = ()


class ComprehensionBody(ast.AST):
    _attributes = ()
    _fields = ("value", "collection")

    def __init__(self, value, collection):
        self.collection = collection
        self.value = value


class RewriteComprehensionArgs(ast.NodeTransformer):
    def __init__(self, args):
        self.args = args
        self.cls = None

    def visit_Name(self, node):
        if node.id.startswith("."):
            var_idx = int(node.id[1:])
            return self.args[var_idx]
        return node

    def visit_For(self, node: ast.For):
        assert (
            len(node.body) == 1
            or len(node.body) == 2
            and isinstance(node.body[1], ast.Continue)
        )
        target = self.visit(node.target)
        elt = self.visit(node.body[0])
        iter = self.visit(node.iter)
        condition = None
        if isinstance(elt, ast.If):
            condition = elt.test
            assert (
                len(elt.body) == 1
                or len(elt.body) == 2
                and isinstance(elt.body[1], ast.Continue)
            )
            elt = elt.body[0]
        generators = [
            ast.comprehension(
                target=target,
                iter=iter,
                ifs=[condition] if condition is not None else [],  # TODO
                is_async=False,  # TODO
            )
        ]
        # if not isinstance(elt, ast.IfExp):
        #    raise Exception("Expected IfExp instead of " + ast.dump(elt))
        # TODO handle DictComp
        if self.cls and isinstance(elt, self.cls):
            generators = generators + elt.generators
            elt = elt.elt
        elif isinstance(elt, ComprehensionBody):
            self.cls = {
                ast.List: ast.ListComp,
                ast.Set: ast.SetComp,
                ast.Dict: ast.DictComp,
            }[type(elt.collection)]
            elt = elt.value
        elif isinstance(elt, ast.Expr) and isinstance(elt.value, ast.Yield):
            self.cls = ast.GeneratorExp
            elt = elt.value.value
        else:
            raise Exception("Unexpected " + ast.dump(elt) + ", cls:" + str(self.cls))

        if issubclass(self.cls, ast.DictComp):
            return ast.DictComp(
                key=elt[0],
                value=elt[1],
                generators=generators,
                ifs=[],
            )
        else:
            return self.cls(
                elt=elt,
                generators=generators,
                ifs=[],
            )


class RemoveLastContinue(ast.NodeTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_remove_continue = False

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if field in ("body", "orelse") and isinstance(old_value, list):
                new_body = []
                last_val = self.can_remove_continue
                for last_in_body, split in [
                    (False, old_value[:-1]),
                    (True, old_value[-1:]),
                ]:
                    self.can_remove_continue = (
                        (isinstance(node, (ast.For, ast.While)) or last_val)
                        and last_in_body
                        and not (
                            len(old_value) == 1
                            and isinstance(old_value[0], ast.Continue)
                        )
                    )
                    for value in split:
                        if isinstance(value, ast.AST):
                            value = self.visit(value)
                            if value is None:
                                continue
                        new_body.append(value)
                self.can_remove_continue = last_val
                old_value[:] = new_body
            elif isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                    new_values.append(value)
                old_value[:] = new_values
        return node

    def visit_Continue(self, node):
        if self.can_remove_continue:
            return None
        return node


class WhileBreakFixer(ast.NodeTransformer):
    def __init__(self, while_fusions):
        self.while_fusions = while_fusions

    def visit_Break(self, node):
        if hasattr(node, "_loop_node"):
            if node._loop_node in self.while_fusions:
                return ast.Continue()
        return node


def negate(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return node.operand
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # num_neg = sum(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) for n in node.values)
        # if num_neg > len(node.values) / 2:
        return ast.BoolOp(op=ast.And(), values=[negate(n) for n in node.values])
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        # num_neg = sum(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) for n in node.values)
        # if num_neg > len(node.values) / 2:
        return ast.BoolOp(op=ast.Or(), values=[negate(n) for n in node.values])
    return ast.UnaryOp(op=ast.Not(), operand=node)


def get_origin(trees: ast.AST):
    origin = None
    offset = None
    if isinstance(trees, ast.AST):
        trees = [trees]
    for tree in trees:
        for node in ast.walk(tree):
            try:
                if offset is None or node._origin.offset < offset:
                    origin = node._origin_node
            except AttributeError:
                pass
        return origin


def walk_with_parent(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque

    todo = deque([(None, node)])
    while todo:
        parent, node = todo.popleft()
        todo.extend((node, child) for child in ast.iter_child_nodes(node))
        yield parent, node


def remove_from_parent(item: ast.AST, parent: ast.AST):
    for field, old_value in ast.iter_fields(parent):
        if isinstance(old_value, list):
            if item in old_value:
                old_value.remove(item)
                return True
        elif old_value is item:
            delattr(parent, field)
            return True
    return False


def make_bool_op(op, values):
    assert len(values) > 0 and isinstance(op, (ast.And, ast.Or))
    new_values = []
    for v in values:
        if isinstance(v, ast.BoolOp) and isinstance(v.op, op.__class__):
            new_values.extend(v.values)
        else:
            new_values.append(v)
    return ast.BoolOp(op=op, values=new_values)


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
