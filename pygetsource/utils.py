import ast
import dis
import sys
from typing import TypeVar

try:
    # Try/except to skip circular import errors
    # but benefit from type checking
    from .decompiler import Node
except ImportError:
    Node = TypeVar("Node")

binop_to_ast = {
    "BINARY_MATRIX_MULTIPLY": ast.MatMult(),
    "BINARY_POWER": ast.Pow(),
    "BINARY_MULTIPLY": ast.Mult(),
    "BINARY_MODULO": ast.Mod(),
    "BINARY_ADD": ast.Add(),
    "BINARY_SUBTRACT": ast.Sub(),
    "BINARY_FLOOR_DIVIDE": ast.FloorDiv(),
    "BINARY_TRUE_DIVIDE": ast.Div(),
    "BINARY_LSHIFT": ast.LShift(),
    "BINARY_RSHIFT": ast.RShift(),
    "BINARY_AND": ast.BitAnd(),
    "BINARY_XOR": ast.BitXor(),
    "BINARY_OR": ast.BitOr(),
}

if sys.version_info >= (3, 11):
    nb_ops_map = {sign: i for i, (name, sign) in enumerate(dis._nb_ops)}

    nb_inplace_ops_to_ast = {
        nb_ops_map["+="]: ast.Add(),
        nb_ops_map["&="]: ast.BitAnd(),
        nb_ops_map["//="]: ast.FloorDiv(),
        nb_ops_map["<<="]: ast.LShift(),
        nb_ops_map["@="]: ast.MatMult(),
        nb_ops_map["*="]: ast.Mult(),
        nb_ops_map["%="]: ast.Mod(),
        nb_ops_map["|="]: ast.BitOr(),
        nb_ops_map["**="]: ast.Pow(),
        nb_ops_map[">>="]: ast.RShift(),
        nb_ops_map["-="]: ast.Sub(),
        nb_ops_map["/="]: ast.Div(),
        nb_ops_map["^="]: ast.BitXor(),
    }

    nb_binary_ops_to_ast = {
        nb_ops_map["+"]: ast.Add(),
        nb_ops_map["&"]: ast.BitAnd(),
        nb_ops_map["//"]: ast.FloorDiv(),
        nb_ops_map["<<"]: ast.LShift(),
        nb_ops_map["@"]: ast.MatMult(),
        nb_ops_map["*"]: ast.Mult(),
        nb_ops_map["%"]: ast.Mod(),
        nb_ops_map["|"]: ast.BitOr(),
        nb_ops_map["**"]: ast.Pow(),
        nb_ops_map[">>"]: ast.RShift(),
        nb_ops_map["-"]: ast.Sub(),
        nb_ops_map["/"]: ast.Div(),
        nb_ops_map["^"]: ast.BitXor(),
    }
else:
    nb_ops_map = {}
    nb_inplace_ops_to_ast = {}
    nb_binary_ops_to_ast = {}

unaryop_to_ast = {
    "UNARY_POSITIVE": ast.UAdd(),
    "UNARY_NEGATIVE": ast.USub(),
    "UNARY_NOT": ast.Not(),
    "UNARY_INVERT": ast.Invert(),
}

inplace_to_ast = {
    "INPLACE_MATRIX_MULTIPLY": ast.MatMult(),
    "INPLACE_FLOOR_DIVIDE": ast.FloorDiv(),
    "INPLACE_TRUE_DIVIDE": ast.Div(),
    "INPLACE_ADD": ast.Add(),
    "INPLACE_SUBTRACT": ast.Sub(),
    "INPLACE_MULTIPLY": ast.Mult(),
    "INPLACE_MODULO": ast.Mod(),
    "INPLACE_POWER": ast.Pow(),
    "INPLACE_LSHIFT": ast.LShift(),
    "INPLACE_RSHIFT": ast.RShift(),
    "INPLACE_AND": ast.BitAnd(),
    "INPLACE_XOR": ast.BitXor(),
    "INPLACE_OR": ast.BitOr(),
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

hasjabs = [
    "JUMP_ABSOLUTE",
    "FOR_ITER",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "JUMP_FORWARD",
    "JUMP_BACKWARD",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_TRUE_OR_POP",
    "JUMP_IF_NOT_EXC_MATCH",
    "SETUP_FINALLY",
    "SEND",
]

no_ops = [
    "NOP",
    "POP_BLOCK",
    "SETUP_LOOP",
    "RESUME",
    "CACHE",
    "PRECALL",
    "PUSH_NULL",
    "MAKE_CELL",
    "COPY_FREE_VARS",
]


def graph_sort(root):
    """
    Topological sort of graph

    Parameters
    ----------
    root: Node
        The root node of the graph to sort

    Returns
    -------
    list[Node]
        Keys of the graph in topological order
    """

    visited = set()
    order = []

    def dfs(block):
        if block in visited:
            return
        visited.add(block)
        if block.next:
            dfs(block.next)
        for jump in block.jumps:
            dfs(jump)
        order.append(block)

    # TODO handle disjoint components
    dfs(root)
    return order[::-1]
