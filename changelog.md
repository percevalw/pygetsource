## Unreleased

- Added partial Python 3.12 decompilation support (new opcode/jump encodings such as `RETURN_CONST`, `COMPARE_OP`, `POP_JUMP_*`, `FOR_ITER`, and `END_FOR` handling).
- Improved reconstruction for conditional expressions to account for specific control-flow artifacts in generated source.
- Added recovery of inline Python 3.12 comprehensions (filters and nested) back to `list`/`set`/`dict` comprehensions.

## v0.4.0

- Stable baseline focused on pre-3.12 bytecode patterns.
- Python 3.12 support was limited/incomplete.
