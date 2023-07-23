# PyGetSource

PyGetSource is a Python decompiler, aiming to convert bytecode instructions back into Python code. This is useful when trying to recover the source code of a function from its bytecode instructions.

## Overview

PyGetSource is a Python decompiler. When Python reads code, it first converts the instructions into bytecode. For instance:

For instance:
```python
a = 2
```

is converted into

```
LOAD_CONST 0
STORE_FAST 1
```

The latter form is typically stored in `.pyc` files and in the `__code__` attribute of function objects. PyGetSource's goal is to reverse this process.

PyGetSource is still in development. It should be able to recover the source code of simple functions for various programs from Python 3.7 to Python 3.10. It is not yet able to recover the source code of classes, import statements, try/except blocks, and does not support Python 2. While functional, the codebase has not been optimized and is in need of significant refactoring.

## When is this useful ?

PyGetSource proves useful when you need to recover the source code from a `.pyc` file, or when a function is created through an eval statement or lambda syntax. Python versions earlier than 3.11 do not store the exact character offset of the function in the source code, making this tool particularly helpful in those cases.

PyGetSource is inspired by the `inspect.getsource` function, which retrieves the source code of a function. However, it's not always applicable, as noted above.

The project takes its name from the `inspect.getsource` function, which returns the source code of a function. This function is not always applicable, as explained above.

## Alternatives

[uncompyle6](https://github.com/rocky/python-uncompyle6) is a Python decompiler that supports Python 2 and 3 up to Python 3.8. It uses a grammar-based approach to rebuild code from bytecode patterns. While it's effective for Python versions up to 3.9, it is less effective for higher versions that introduce various optimizations. It also employs a copyleft GPL license, making it less suitable for larger projects with permissive licenses.

[decompyle++ (pycdc)](https://github.com/zrax/pycdc) uses a state machine approach to build an AST iteratively by processing bytecode instructions. It's written in C++ and supports more Python versions than uncompyle6. However, it also uses a copyleft license.

## How does it work ?

PyGetSource uses a distinct approach. The bytecode instructions are initially converted into a directed graph, representing the program's flow. This graph is then iteratively reduced, processing each node based on its opcode, argument, and position. This method allows us to rely more on high-level patterns and less on Python’s unique features when recreating complex structures like nested loops and break/return statements.

Here is an example of a graph being reduced:

![Graph reduction](./docs/graph-example.svg)


PyGetSource uses the `ast` and `astunparse` libraries to generate the source code, instead of creating this functionality from scratch.

Finally, this software is distributed under the permissive MIT license, making it ideal for use as a dependency in larger projects.

## When is a decompilation successful ?

Since the compilation process is injective, it's impossible to recover the exact original source code. Multiple Python programs can yield the same bytecode instructions. Also, the original source code is typically unavailable.

If we recompile the generated program, we can compare the two sets of bytecode instructions to ensure functional equivalence. However, Python may introduce 'no-op' codes (like 'NOP') that might cause this verification to fail despite the two code objects being functionally equivalent.

Instead, PyGetSource compares the graph of the original code object with the graph of the recovered code object, after a pruning step. During this step, 'no-op' codes are removed, jump instructions are pruned (while maintaining edges between source and target nodes), and dead-code is eliminated.


