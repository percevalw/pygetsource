import dis
import io
import sys
from typing import List, Optional, Set
import ast
import textwrap

from IPython.display import display

from .utils import (
    hasjabs,
    graph_sort,
    get_origin_offset,
    lowest_common_successor, detect_loops,
)

try:
    import ast as astunparse

    assert hasattr(astunparse, "unparse")
    print("Using ast.unparse")
except AssertionError:
    import astunparse as astunparse
from types import CodeType

INDENT = 0
DEBUG = True


def debug(*args, **kwargs):
    if DEBUG:
        buffer = io.StringIO()
        print(*args, **kwargs, file=buffer)
        print(textwrap.indent(buffer.getvalue(), "  " * INDENT), end="")


class Unpacking(ast.AST):
    _attributes = ()
    _fields = ("value",)

    def __init__(self, value, counter, starred=False):
        self.value = value
        self.counter = counter
        self.starred = starred

    def __repr__(self):
        return f"Unpacking({self.value}, {self.counter}, starred={self.starred})"


class Node:
    def __init__(self, code: CodeType, op_idx: int, offset: int, opname: str, arg: int):
        # Two choices when walking the graph, either go to the next node or
        # jump to a different node in case of a (un)conditional jump
        self.next: Optional[Node] = None
        self.jumps: Set[Node] = set()

        # Previous nodes
        self.prev: Set[Node] = set()

        # The node's code
        self.code = code
        self.opname = opname
        self.arg = arg
        self.op_idx = op_idx
        self.offset = offset

        self.stmts: List[ast.AST] = []
        self.stack: List[ast.AST] = []

        # The node's position in the topological sort
        self.index: int = -1
        self.visited: bool = False
        self.loops = set()

    def add_stmt(self, stmt):
        debug("Add stmt", self, ":", stmt)
        debug(
            textwrap.indent(astunparse.unparse(ast.fix_missing_locations(stmt)), "  ")
        )
        self.stmts.append(stmt)

    def add_stack(self, item):
        debug("Add stack", self, ":", item)
        self.stack.append(item)

    def pop_stack(self):
        return self.stack.pop()

    def contract_backward(self):
        """
        Merge the node with its predecessor
        """

        # If the node has a single predecessor, merge it with that node

        prev_node = next((n for n in self.prev if n.next is self), None)
        if not self.prev:
            self.loops = frozenset()
        if not (
            prev_node
            and (
                (len(self.prev) == 1 and not prev_node.jumps)
                or prev_node.opname == "NOP"
            )
        ):
            return None

        debug(
            "Contract backward", prev_node, "<<<", self, "|", self.next, self.jumps, "|", self.prev,
        )
        self.prev.remove(prev_node)
        for pred_pred in prev_node.prev:
            if pred_pred.next is prev_node:
                pred_pred.next = self
            pred_pred.jumps = {
                self if jump is prev_node else jump for jump in pred_pred.jumps
            }
            self.prev.add(pred_pred)

        self.jumps = {
            (self if n is prev_node else n) for n in (self.jumps | prev_node.jumps)
        }
        self.stmts = prev_node.stmts + self.stmts
        debug("After contraction", self, "|", self.next, self.jumps, "|", self.prev)

        return prev_node

    @classmethod
    def from_code(cls, code: CodeType):
        bytes = code.co_code
        offset = 0
        graph = {}

        while offset < len(bytes):
            op_idx = offset
            op, arg = bytes[offset], bytes[offset + 1]
            opname = dis.opname[op]
            offset += 2

            ex_arg_count = 0
            ex_arg = 0
            while opname == "EXTENDED_ARG":
                ex_arg = arg << (8 * (ex_arg_count + 1))
                op, arg = bytes[offset], bytes[offset + 1]
                opname = dis.opname[op]
                offset += 2
            arg += ex_arg

            if sys.version_info >= (3, 10) and opname in hasjabs:
                arg *= 2

            block = cls(code, op_idx, offset, opname=opname, arg=arg)
            for i in range(op_idx, offset, 2):
                graph[i] = block

        # graph[offset] = Block(code, offset, offset + 2, opname="END", arg=0, graph=graph)

        for block in graph.values():
            opname = block.opname
            arg = block.arg

            if opname == "RETURN_VALUE":
                pass
            elif opname == "JUMP_FORWARD":
                block.jumps.add(graph[block.op_idx + 2 + arg])
            elif opname == "JUMP_ABSOLUTE":
                block.jumps.add(graph[arg])
            elif opname in (
                "POP_JUMP_IF_TRUE",
                "POP_JUMP_IF_FALSE",
            ):
                block.next = graph[block.op_idx + 2]
                block.jumps.add(graph[arg])
            elif opname in ("FOR_ITER",):
                block.next = graph[block.op_idx + 2]
                block.jumps.add(graph[block.op_idx + 2 + arg])
            else:
                block.next = graph[block.op_idx + 2]

            if block.next:
                block.next.prev.add(block)
            for jump in block.jumps:
                jump.prev.add(block)

        for index, node in enumerate(graph_sort(graph)):
            node.index = index

        detect_loops(graph[0])

        return graph[0]

    def __repr__(self):
        return f"[{self.op_idx}]({self.opname}, {self.arg})"

    def draw(self):
        from IPython.core.display import SVG
        import networkx as nx

        g = nx.DiGraph()

        visited = set()
        positions = {}

        def rec(node: Node, pos):
            if node in visited:
                return
            visited.add(node)
            source = node.to_source().split("\n")
            max_line = max(len(line) for line in source)
            label = "\n".join([
                f"[{node.op_idx}]{'✓' if node.visited else ''}|{len(node.stack)}☰|{len(node.prev)}↣",
                f"↺({','.join(str(i) for i in sorted(node.loops))})",
                f"{node.opname}({node.arg})",
                "------------",
                *[line.ljust(max_line) for line in source],
            ])
            g.add_node(
                node,
                pos=",".join(map(str, pos)) + "!",
                label=label,
                fontsize=10,
            )
            # positions[node] = pos
            if node.next:
                g.add_edge(node, node.next, label="next", fontsize=10)
                rec(node.next, (pos[0], pos[1] + 1))

            for jump in node.jumps:
                g.add_edge(node, jump, label="jump", fontsize=10)
                rec(jump, (pos[0], pos[1] + 1))

            for p in node.prev:
                rec(p, (pos[0], pos[1] - 1))

        rec(self, (0, 0))
        agraph = nx.nx_agraph.to_agraph(g)
        # Update some global attrs like LR rankdir, box shape or monospace font
        agraph.graph_attr.update(rankdir="LR", fontname="Menlo")
        agraph.node_attr.update(shape="box", fontname="Menlo")
        agraph.edge_attr.update(fontname="Menlo")
        svg = agraph.draw(prog="dot", format="svg")
        return SVG(svg)
        # return nx.draw(
        #     g,
        #     pos=positions,
        #     font_size=8,
        #     with_labels=True,
        #     horizontalalignment="left",
        # )

    def to_source(self):
        res = []
        for stmt in self.stmts:
            try:
                stmt = ast.fix_missing_locations(stmt)
            except AttributeError:
                pass
            res.append(astunparse.unparse(stmt))
        return "\n".join(res)

    # noinspection PyMethodParameters,PyMethodFirstArgAssignment
    def run(node: "Node", stop_nodes: Set["Node"] = frozenset(), upper_loops: Set["Node"] = frozenset()):
        """
        Convert the graph into an AST
        """
        node: Node

        if node.visited or node in stop_nodes:
            return None

        while True:
            debug("::: Processing", node, "|", node.next, node.jumps, "|", node.prev)
            node.visited = True
            prev_visited = [n for n in node.prev if n.visited]
            prev_unvisited = [n for n in node.prev if not n.visited]
            if len(prev_visited) > 1:
                debug("TOO BAD :( too many prev visited", prev_visited, "for", node)
                break
            if prev_visited:
                node.stack = list(prev_visited[0].stack)

            if node.opname.startswith("LOAD_"):
                process_binding_(node)
            elif node.opname.startswith("STORE_"):
                process_store_(node)
            elif node.opname == "RETURN_VALUE":
                node.add_stmt(ast.Return(node.pop_stack()))
            elif node.opname == "JUMP_FORWARD":
                loops = upper_loops | node.loops
                jump = next(iter(node.jumps))
                # If we don't leave any loop
                if loops and (jump.loops & loops) != loops:
                    node.add_stmt(ast.Break())
            elif node.opname == "JUMP_ABSOLUTE":
                debug("LOOOOOPS", upper_loops | node.loops, node, node.jumps)
                loops = upper_loops | node.loops
                jump = next(iter(node.jumps))
                # If we don't leave any loop
                if loops and (jump.loops & loops) == loops:
                    if jump.visited:
                        node.add_stmt(ast.Continue())
                else:
                    node.add_stmt(ast.Break())
            elif node.opname in ("POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"):
                if_node = node.next
                [else_node] = node.jumps
                meet_nodes = lowest_common_successor(
                    if_node,
                    else_node,
                    stop_nodes={*stop_nodes, node},
                )
                meet_nodes = {n for n in meet_nodes}
                debug("IF/ELSE BLOCK", node, "MEET", meet_nodes, "(STOP at", {*stop_nodes, node}, ")")
                assert len(meet_nodes) <= 1
                meet_node = next(iter(meet_nodes), None)

                before = (if_node, else_node)
                if_node = if_node.run(
                    stop_nodes={*stop_nodes, *meet_nodes, node},
                    upper_loops=(upper_loops | node.loops),
                )
                else_node = else_node.run(
                    stop_nodes={*stop_nodes, *meet_nodes, node},
                    upper_loops=(upper_loops | node.loops)
                )
                debug("IF/ELSE", node, "MEET", meet_nodes)
                debug("IF/ELSE", node, "BEFORE succ", before, "-> AFTER succ", if_node, else_node)

                node.add_stmt(
                    ast.If(
                        test=(
                            node.pop_stack()
                            if node.opname == "POP_JUMP_IF_FALSE"
                            else ast.UnaryOp(
                                op=ast.Not(),
                                operand=node.pop_stack(),
                            )
                        ),
                        body=if_node.stmts if if_node else [],
                        orelse=else_node.stmts if else_node else [],
                    )
                )

                # noinspection PyTypeChecker
                successors: Set[Node] = {
                    *((if_node.next, *if_node.jumps) if if_node else (node.next,)),
                    *((else_node.next, *else_node.jumps) if else_node else node.jumps),
                } - {
                    None
                }  # type: ignore
                unvisited_successors = [n for n in successors if not n.visited]

                # Forward contraction
                #if meet_node:
                #    next_node = meet_node
                #    jump_nodes = successors - {meet_node}
                # If only one successor that doesn't make us leave a loop
                # then tag it as the next node
                # elif (
                #       len(unvisited_successors) == 1 and
                #       (unvisited_successors[0].loops & loops) == loops
                # ):
                #     debug(
                #         "IF/ELSE",
                #         node,
                #         "has only one unvisited successor",
                #         unvisited_successors,
                #     )
                #     next_node = unvisited_successors[0]
                #     jump_nodes = {n for n in successors if n.visited}
                #else:
                debug("No next node, only jumps to", successors)
                next_node = None
                jump_nodes = successors
                for succ in successors:
                    debug("IF/ELSE", node, ": REBINDING SUCCESSOR", succ, "=>", succ.prev, "->", succ.prev - {if_node, else_node} | {node})
                    succ.prev = (succ.prev - {if_node, else_node}) | {node}
                node.next = next_node
                node.jumps = jump_nodes
                # debug("IF/ELSE", node, "FINAL LOOPS", if_node.loops, else_node.loops, "=", node.loops)
                display(node.draw())

            while True:
                prev = node.contract_backward()
                if not prev:
                    break
                if node in node.jumps:
                    node.jumps.remove(node)
                    node.prev.remove(node)
                    body = node.stmts
                    node.stmts = []
                    node.add_stmt(
                        ast.While(
                            test=ast.Constant(value=True, kind=None),
                            body=body,
                            orelse=[],
                        )
                    )
                    node.loops = upper_loops
                display(node.draw())

            display(node.draw())
            # if node.jumps:
            #     debug("JUMPS !!", node.jumps)

            debug("::: done", node, "|", node.next, node.jumps, "|", node.prev)
            if not node.next:
                loops = upper_loops | node.loops
                candidate_jumps = [
                    n for n in node.jumps
                    if not n.visited
                    and (n.loops & loops) == loops
                ]
                debug("CANDIDATE JUMPS", node, "=>", candidate_jumps, (upper_loops, node.loops))
                assert len(candidate_jumps) <= 1, candidate_jumps
                if candidate_jumps and candidate_jumps[0] not in stop_nodes:
                    node.next = candidate_jumps[0]
                    node.jumps.remove(node.next)


            if node.next and not node.next.visited and not node.next in stop_nodes:
                node: Node = node.next
            else:
                break

        return node


def process_binding_(node):
    # assert len(block.pred) <= 1
    opname = node.opname
    arg = node.arg
    code = node.code
    idx = node.op_idx
    opname = opname.split("_")[1]

    if opname == "FAST":
        node.add_stack(
            ast.Name(
                id=code.co_varnames[arg],
                _origin_offset=idx,
            )
        )
    elif opname in ("NAME", "GLOBAL"):
        node.add_stack(
            ast.Name(
                id=code.co_names[arg],
                _origin_offset=idx,
            )
        )
    elif opname == "DEREF":
        node.add_stack(
            ast.Name(
                id=(code.co_cellvars + code.co_freevars)[arg],
                _origin_offset=idx,
            )
        )
    elif opname == "CONST":
        const_value = code.co_consts[arg]
        if isinstance(const_value, frozenset):
            const_value = set(const_value)
        node.add_stack(
            ast.Constant(
                value=const_value,
                kind=None,
                _origin_offset=idx,
            )
        )
    # ATTRIBUTES
    elif opname in ("ATTR", "METHOD"):
        value = node.pop_stack()
        node.add_stack(
            ast.Attribute(
                value=value,
                attr=code.co_names[arg],
                _origin_offset=idx,
            )
        )
    elif opname in ("SUBSCR",):
        value = node.pop_stack()
        node.add_stack(
            ast.Subscript(
                slice=value,
                value=node.pop_stack()[1],
                _origin_offset=idx,
            )
        )
    else:
        raise ValueError("Unknown opname", opname)


def process_store_(node: Node):
    process_binding_(node)
    target = node.pop_stack()
    value = node.pop_stack()
    idx = node.op_idx
    if isinstance(value, ast.AugAssign):
        node.add_stmt(value)
    elif isinstance(value, ast.FunctionDef) and isinstance(target, ast.Name):
        node.add_stmt(value)
    else:
        unpacking = value if isinstance(value, Unpacking) else None
        if unpacking:
            value = value.value
            target = ast.Starred(target) if unpacking.starred else target
            multi_targets = [ast.Tuple([target])]
        else:
            multi_targets = [target]

        if (
            len(node.stmts)
            and isinstance(node.stmts[-1], ast.Assign)
            and idx <= get_origin_offset(target)
        ):
            prev = node.stmts.pop()
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
                or (unpacking and len(prev_multi_targets[-1].elts) > unpacking.counter)
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
                        value = ast.Tuple([*prev.value.elts, value], ctx=ast.Load())
                    else:
                        value = ast.Tuple([prev.value, value], ctx=ast.Load())
                # value is value

        node.add_stmt(
            ast.Assign(
                targets=multi_targets,
                value=value,
            ),
            # start=op_idx,
        )
