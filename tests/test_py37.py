import pytest as pytest
from utils import make_test_idem


@make_test_idem
def test_assign():
    a = 1


@make_test_idem
def test_attr_assign():
    a = 1
    a.b = 2
    a.b, a.c = 3, 4


@make_test_idem
def test_subscr_assign():
    a = 1
    a[0] = 2
    a[b], a[c] = 3, 4


@make_test_idem
def test_delete():
    del a


@make_test_idem
def test_delete_attr():
    del a.b


@make_test_idem
def test_delete_subscr():
    del a[0]


@make_test_idem
def test_store():
    a = 1
    b = a


@make_test_idem
def test_store_attr():
    a = 1
    b = a.b


@make_test_idem
def test_store_subscr():
    a = 1
    b = a[0]


@make_test_idem
def test_if():
    a = 1
    if a == 1:
        a = 2


@make_test_idem
def test_empty_if():
    a = 1
    if a == 1:
        pass


@make_test_idem
def test_if_not():
    a = 1
    if not a == 1:
        a = 2


@make_test_idem
def test_if_else():
    a = 1
    if a == 1:
        a = 2
    else:
        a = 3


@make_test_idem
def test_ternary():
    a = 5 if x > 0 else 6
    print(5 if x > 0 else 6 if x < 0 else 7)
    u = {
        0: "ok" if y == x else "ko",
        1: "ko" if y == x else "ok",
    }


@make_test_idem
def test_while():
    a = 1
    while a == 1:
        a = 2


@make_test_idem
def test_while_break():
    a = 1
    while a == 1:
        a = 2
        break
        a = 4


@make_test_idem
def test_while_if():
    a = 1
    while test:
        if a == 2:
            x = 2
        elif a == 3:
            x = 4


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_while_try_except_break():
    while test:
        try:
            x = 2
            break
        except:
            break


@make_test_idem
def test_nested_while():
    a = 1
    while a == 1:
        while b == 2:
            a = 2


@make_test_idem
def test_infinite_while():
    while True:
        if a == 0:
            x = 1
        elif b:
            x = 2
        else:
            x = 3
    print("ok")


@make_test_idem
def test_infinite_while_if_break():
    while True:
        if a == 0:
            continue
        elif b:
            break


@make_test_idem
def test_for():
    a = 1
    for i in range(10):
        a = 2


@make_test_idem
def test_for_if():
    for j in range(10):
        if test:
            a = 4
        else:
            a = 3


@make_test_idem
def test_for_if_return():
    for j in range(10):
        if test:
            a = 4
            return z
        else:
            a = 3


@make_test_idem
def test_for_if_continue():
    for i in collection:
        if test:
            a = 4
        else:
            continue
            a = 3


@make_test_idem
def test_nested_for():
    a = 1
    for i in range(10):
        for j in range(10):
            a = 3

    a = 8
    for i in range(10):
        continue
        for j in range(10):
            if j % 2 == 0:
                break
            else:
                a = 3
        break


@make_test_idem
def test_for_break():
    a = 1
    for i in range(10):
        a = 2
        break
        a = 4


@make_test_idem
def test_for_break_in_single_if():
    a = 1
    for i in range(10):
        if i == 0:
            a = 2
            break


@make_test_idem
def test_for_break_in_single_for():
    a = 1
    for i in range(10):
        for i in range(10):
            a = 2
            if b == 12:
                break


@make_test_idem
def test_for_continue_hell():
    a = 1
    for i in range(10):
        continue
        if i == 0:
            a = 3
            continue
        else:
            b = 5
        continue


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_except():
    try:
        a = 1
    except:
        a = 2


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_except_else():
    try:
        a = 1
    except:
        a = 2
    else:
        a = 3


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_except_finally():
    try:
        a = 1
    except:
        a = 2
    finally:
        a = 3

    z = 4


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_except_else_finally():
    try:
        a = 1
    except:
        a = 2
    else:
        a = 3
    finally:
        a = 4


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_typed_except():
    try:
        a = 1
    except StopIteration as e:
        a = 2


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_typed_except_finally():
    try:
        a = 1
    except StopIteration as e:
        a = 2
    finally:
        a = 3

    z = 4


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_typed_except_else_finally():
    try:
        a = 1
    except StopIteration:
        a = 2
    else:
        a = 3
    finally:
        a = 4


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_multi_typed_except():
    try:
        a = 1
    except StopIteration:
        a = 2
    except ValueError:
        a = 3


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_in_loop():
    for x in [1, 2, 3]:
        try:
            try:
                func(2)
            except:
                ok = 1
            else:
                ok = 2
            finally:
                ok = 3
        except:  # noqa
            break


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_try_in_infinite_loop():
    while True:
        try:
            try:
                func(2)
            except:
                ok = 1
            else:
                ok = 2
            finally:
                ok = 3
        except:  # noqa
            break


@make_test_idem
def test_gen_comp():
    return (a for a in range(10))


@make_test_idem
def test_list_comp():
    return [a for a in range(10)]


@make_test_idem
def test_set_comp():
    return {i for a in range(10) for i in range(a)}


@make_test_idem
def test_dict_comp():
    return {a: b for a, b in enumerate(range(10))}


@make_test_idem
def test_dict_comp_ex():
    return {a: b for (*a, b) in enumerate(range(10))}


@make_test_idem
def test_comp_if():
    return [a for a in range(10) if a % 2 == 0]


@make_test_idem
def test_comp_if_else():
    return [a if a % 2 == 0 else 0 for a in range(10)]


@make_test_idem
def test_multi_comp():
    return [b for a in range(10) for b in range(10)]


@make_test_idem
def test_comp_nested():
    u = 1
    return [[u for a in b] for b in range(10)]


@make_test_idem
def test_method():
    return a.method()


@make_test_idem
def test_build_collections():
    a = [1, 2, 3]
    b = (1, 2, 3)
    c = {1, 2, 3}
    d = {1: 2, 3: a}
    e = (a, b, c, d)
    f = {a: b, c: d}


@make_test_idem
def test_build_collections_ex():
    a = [b, *c, *d, e]
    a = {b, *c, *d, e}
    a = (b, *c, *d, e)
    a = {b: c, **d, e: 1}


@make_test_idem
def test_build_collections_unpack_map():
    a = {**{1: 1, 2: 2, 3: 3}, "e": 1, "f": 2}


@make_test_idem
def test_binary_ops():
    a = b + 2
    a = b - 2
    a = b * 2
    a = b / 2
    a = b**2
    a = b // 2
    a = b @ 2
    a = b % 2
    a = b | 2
    a = b & 2
    a = b ^ 2
    a = b << 2
    a = b >> 2
    return None


@make_test_idem
def test_contains():
    a = 1 in b
    a = 1 not in b


@make_test_idem
def test_is():
    a = 1 is b
    a = 1 is not b


@make_test_idem
def test_inplace_ops():
    a @= b
    a //= b
    a /= b
    a += b
    a -= b
    a *= b
    a %= b
    a **= b
    a <<= b
    a >>= b
    a &= b
    a ^= b
    a |= b


@make_test_idem
def test_unary_ops():
    a = +b
    a = -b
    a = ~b
    a = not b


def make_test_deref():
    x = 1

    def func():
        return x

    return func


test_deref = make_test_idem(make_test_deref())


@make_test_idem
def test_print():
    print("test")


@make_test_idem
def test_const():
    5


@make_test_idem
def test_format():
    a = 1
    b = 2
    return f"ceci {a!s} est {b!r} un {c!a}: {u:.2f}"


@make_test_idem
def test_comprehension_format():
    a = 1
    b = 2
    return f"list: {[i for j in a]!s}"


@make_test_idem
def test_slice():
    b = a[1:2:3]
    b[::-1] = 1
    a[1:2]


@make_test_idem
def test_unpack():
    a, b = z
    a, b = [1, 2]


@make_test_idem
def test_unpack_star():
    a, *b = z
    a, *b, c, d = z


@make_test_idem
def test_call():
    a = b(1, 2, 3)
    a = b(1, 2, c=3)
    a = b(u=2, c=3)


@make_test_idem
def test_call_ex():
    a = b(1, 2, *c, **d)
    a = b(1, 2, *c, **d, e=1, f=2)
    z = b(1, 2, *c, **{1: 1, 2: 2, 3: 3}, e=1, f=2)


@make_test_idem
def test_call_dict_comp_kwargs():
    a = b(
        1,
        2,
        **{str(i): i for i in range(12)},
        **d,
    )


@make_test_idem
def test_swap():
    a, b = b, a
    a, b, c = b, c, a
    a, b, c, d = b, c, d, a


@make_test_idem
def test_multi_assign():
    a = b = c = 1
    a, b = c = d, e = [1, 2]
    a, b = (*c,) = *d, e = [1, 2]


@make_test_idem
def test_complex_swap(x):
    x.y, x.z = x.z, x.y
    x[0], x[1] = x[1], x[0]
    x[0].y, x[0].z = x.z[1], x.z[1]


@make_test_idem
def test_if_in_while():
    while test:
        if i == 0:
            x = 1
        elif i == 1:
            x = 0
        else:
            i = 3


@make_test_idem
def test_if_in_while_continue():
    while test:
        if i == 0:
            x = 1
        elif i == 1:
            x = 0
            continue
        else:
            i = 3
            continue


@make_test_idem
def test_if_in_while_break():
    while test:
        if i == 0:
            x = 1
        elif i == 1:
            x = 0
            break
        else:
            i = 3


@make_test_idem
def test_if_in_for():
    for i in test:
        if i == 0:
            x = 1
        elif i == 1:
            x = 0
            break
        else:
            i = 3
            continue


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_break_in_try_in_loop():
    for i in [0]:
        try:
            f(i)
        except:  # noqa
            break


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_break_in_try_finally_in_loop():
    for i in [0]:
        try:
            break
            x = 0
        except:  # noqa
            break
            x = 0
        else:
            break
            x = 0
        finally:
            break
            x = 0


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_return_call_in_try_in_except():
    try:
        try:
            f()
        except:  # noqa
            return f()
    finally:
        return f()


@make_test_idem
def test_iter():
    for x in iter(a):
        pass

    [s2 for s1 in qs.split("&") for s2 in s1.split(";")]


@pytest.mark.xfail(reason="try/except not implemented yet")
@make_test_idem
def test_empty_try():
    try:
        pass
    except:
        pass


@make_test_idem
def test_empty_while():
    while True:
        pass

    while False:
        pass


@make_test_idem
def test_empty_if():
    if True:
        pass

    if False:
        pass


@make_test_idem
def test_nested_function():
    def f():
        def g():
            return 1

        return g()

    return f()


@make_test_idem
def test_while_while_break_return():
    while True:
        if y:
            while True:
                if x:
                    break
                return
            return
        else:
            x = 2


@make_test_idem
def test_while_if_while_if_break_return():
    while True:
        if x:
            while True:
                if x:
                    break
                return None
        else:
            return None


@make_test_idem
def test_while_if_while_if_break_full_return():
    while True:
        if x:
            while True:
                if x:
                    break
                return None
            return None
        else:
            return None


@make_test_idem
def test_while_if_return():
    while True:
        if x:
            return None
        else:
            return None


@make_test_idem
def test_while_if_while_if_break_assign():
    while True:
        if x:
            while True:
                if x:
                    break
                return None
            return None
        x = 2


@make_test_idem
def test_while_only_break():
    a = 1
    while x:
        break
    return None


@pytest.mark.xfail(reason="dead code changes the variable from local to free")
@make_test_idem
def test_while_true_if_if_continue_assign():
    while True:
        if x:
            if y:
                continue
                x = 1
    return z


@make_test_idem
def test_while_true_if_if_continue():
    while True:
        if x:
            if y:
                continue
        return line


@make_test_idem
def test_while_test_if_else_break():
    while test:
        if a:
            x = 2
        else:
            x = 3
            break
    return "a"


@make_test_idem
def test_while_if_if_elif_else():
    while test:
        if a:
            if i == 0:
                x = 1
            elif i == 1:
                x = 0
                break
            else:
                i = 3
                continue

    x = 2
    y = 3
    return "a"


@make_test_idem
def test_while_while_if():
    while z:
        while test:
            if i == 0:
                x = 1
            elif i == 1:
                x = 0
                continue
            else:
                i = 3
                continue

    u = 3
    return u


@make_test_idem
def test_if_and_else():
    if x and y:
        x = 3
    else:
        y = 3

    a = 3
    return a


@make_test_idem
def test_if_and():
    if x and y:
        return 1


@make_test_idem
def test_if_or_else():
    if x or y:
        x = 3
    else:
        y = 3

    a = 3
    return a


@make_test_idem
def test_if_or():
    if x or y:
        return 1


@make_test_idem
def test_if_mix_bool_else():
    if (x or y or z) and (a or b):
        x = 3
    else:
        y = 3


@make_test_idem
def test_if_mix_bool_2_else():
    if (x and y and z) or (a and b):
        x = 3
    else:
        y = 3


@make_test_idem
def test_while_bool():
    while a and b:
        x = 2

    u = 3
    return "u"


@make_test_idem
def test_while_while_bool():
    while a and b:
        while test:
            x = 2

    u = 3
    return "u"


@make_test_idem
def test_while_while_bool_mix():
    while (x and y and z) or (a and b):
        while test:
            x = 2

    u = 3
    return "u"


@make_test_idem
def test_bool_and_of_or_assign():
    z = (x and y and z) or (a and b) or (c and d)


@make_test_idem
def test_bool_or_of_and_assign():
    z = (x or y or z) and (a or b) and (c or d)


@make_test_idem
def test_bool_target_in_comprehension():
    return [(x or y or z) and (a or b) and (c or d) for i in range(10)]


@make_test_idem
async def test_await_simple():
    await z


@make_test_idem
async def test_await_store():
    x = await z


@make_test_idem
def test_yield_from_simple():
    yield from z


@make_test_idem
def test_yield_from_store():
    x = yield from z
