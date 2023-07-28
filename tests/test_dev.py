from pygetsource.decompiler import set_debug, getsource


def make_func(global_var):
    def func():
        a = 1
        while a < 10:
            a = a + 1
            x.method(a, global_var)

        for u in v:
            a, b = b, a

        a, b, c = (a if a % 2 == 0 else 0 for a in b)

    return func


def test_draw():
    func = make_func(1)
    with set_debug(prog="nop"):
        getsource(func.__code__, debug=True)
