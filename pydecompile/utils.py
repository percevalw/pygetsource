import ast
import dis
import functools
import sys
import textwrap
from collections import deque
from typing import List, Set, TypeVar, Dict
try:
    # Try/except to skip circular import errors
    # but benefit from type checking
    from .graph2 import Node
except ImportError:
    pass

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

hasjabs = ["JUMP_ABSOLUTE", "FOR_ITER", "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "JUMP_FORWARD"]



def graph_sort(graph):
    """
    Topological sort of graph

    Parameters
    ----------
    graph: dict[int, Block]
        The graph to sort

    Returns
    -------
    list[Block]
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
    dfs(graph[0])
    return order[::-1]



def get_origin_offset(tree: ast.AST):
    offset = None
    for node in ast.walk(tree):
        try:
            offset = (
                min(offset, node._origin_offset)
                if offset is not None
                else node._origin_offset
            )
        except AttributeError:
            pass
    return offset



def lowest_common_successor(*starts, stop_nodes=None):
    if stop_nodes is None:
        stop_nodes = set()

    queues = [deque([s]) for s in starts]
    visited = [{s} for s in starts]

    while any(queues):
        nodes = [q.popleft() if q else None for q in queues]
        for i, node in enumerate(nodes):
            if not node or node in stop_nodes:
                continue
            for succ in (node.next, *node.jumps):
                if succ and succ not in visited[i] and not succ.visited:
                    queues[i].append(succ)
                    visited[i].add(succ)

        common_nodes = functools.reduce(set.intersection, visited)
        if common_nodes:
            return common_nodes
    return set()


def detect_loops(root: Node):
    """
    Detects all loops in the graph, label them and assign them to the
    `loops` attribute of each node.

    Parameters
    ----------
    root: Node
        The root of the graph
    """

    loop_count = 0
    def explore(node: Node, label: int, ancestors: List[Node], stop_nodes: Set[Node] = set()):
        nonlocal loop_count

        if node in stop_nodes:
            for ancestor in ancestors:
                ancestor.loops = {*ancestor.loops, label}
            return

        if node in ancestors:
            index = ancestors.index(node)
            for ancestor in ancestors[index:]:
                ancestor.loops = {*ancestor.loops, label}
            return
        if node.opname == "FOR_ITER" and node.next not in stop_nodes:
            lca = lowest_common_successor(node.next, next(iter(node.jumps)), stop_nodes={node})
            loop_count += 1
            explore(node.next, loop_count, [node], {*stop_nodes, *lca})
        if node.next:
            explore(node.next, label, ancestors + [node], stop_nodes)
        for jump in node.jumps:
            loop_count += 1
            explore(jump, loop_count, ancestors + [node], stop_nodes)

    explore(root, loop_count, [])