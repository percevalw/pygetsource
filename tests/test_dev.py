import ast

from pygetsource.decompiler import Node, set_debug


def make_func(global_var):
    def func():
        a = 1
        while a < 10:
            a = a + 1
            x.method(a, global_var)

        for u in v:
            a, b = b, a

        a, b, c = (a for a in b)

    return func


def test_draw():
    func = make_func(1)
    root = Node.from_code(func.__code__)
    root.draw(prog="nop")
    with set_debug(prog="nop"):
        root = root.run()
    ast.dump(root.stmts[0])
