import ast

from astunparse.unparser import Unparser


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


def unparse_ForTargetPlaceholder(self, node):
    self.write("[...]")


Unparser._ForTargetPlaceholder = unparse_ForTargetPlaceholder


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


def unparse_ComprehensionBody(self, node):
    """Produced by LIST_APPEND or SET_ADD and unparsed as collection.append/add(value)"""
    self.fill()
    self.write("append/add(")
    self.dispatch(node.value)
    self.write(")")


Unparser._ComprehensionBody = unparse_ComprehensionBody


class RewriteComprehensionArgs(ast.NodeTransformer):
    def __init__(self, args, cls):
        self.args = args
        self.cls = cls

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
            cls = {
                ast.List: ast.ListComp,
                ast.Set: ast.SetComp,
                ast.Dict: ast.DictComp,
            }[type(elt.collection)]
            elt = elt.value
        elif isinstance(elt, ast.Expr) and isinstance(elt.value, ast.Yield):
            cls = ast.GeneratorExp
            elt = elt.value.value
        else:
            raise Exception("Unexpected " + ast.dump(elt) + ", cls:" + str(cls))

        if issubclass(cls, ast.DictComp):
            return ast.DictComp(
                key=elt[0],
                value=elt[1],
                generators=generators,
                ifs=[],
            )
        else:
            return cls(
                elt=elt,
                generators=generators,
                ifs=[],
            )


class RemoveLastContinue(ast.NodeTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_remove_continue = False
        self.current_loops = []

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
                            elif not isinstance(value, ast.AST):
                                new_body.extend(value)
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
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_Continue(self, node):
        if self.can_remove_continue:
            return None
        return node


class WhileBreakFixer(ast.NodeTransformer):
    def __init__(self, while_fusions):
        self.while_fusions = while_fusions

    def visit_Break(self, node):
        if hasattr(node, "_loop_node") and node._loop_node in self.while_fusions:
            return ast.Continue()
        return node
