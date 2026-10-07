# Project documentation

Write all project documentation, usage examples, and docstrings in English.
Preserve upstream license and attribution texts when updating documentation.

Keep retained vendored files under `third_party/` identical to their recorded
upstream snapshots. Record explicitly requested removals in the source manifest.
Place workspace integration changes outside those source directories.

# Type checking

Keep `ty` checks for project interfaces and data structures. Do not add runtime
type checks solely to satisfy the type checker or use `typing.cast`. For a
third-party typing limitation, use a narrowly scoped `ty: ignore[rule]` with a
comment explaining the limitation. Keep runtime validation for optional state
and unsupported inputs, and retain the minimal MuJoCo native API stubs.

# NumPy arrays

Use `np.asarray`, `np.array`, and `astype` only when an actual array conversion,
dtype requirement, or ownership requirement makes them necessary. Normalize
external array-like inputs at the boundary; use existing arrays directly in
internal code rather than repeatedly wrapping them. Do not add conversions
defensively or solely to satisfy the type checker. Preserve intentional copies
that establish ownership or snapshot mutable inputs.

# Runtime validation

Treat this workspace as experimental research code. Favor simple implementation
and debugging observed failures over exhaustive defensive checks.

Validate user configuration at loading boundaries; avoid repeating the same
checks for already validated data throughout internal code. Let NumPy and SciPy
errors propagate instead of catching them only to rewrite their messages.
Avoid speculative guards for extreme numeric values that have not caused a
reproducible problem.

Retain minimal checks for optional or uninitialized state, unsupported inputs,
basic numerical preconditions such as positive durations, and shape mismatches
that would otherwise silently broadcast into incorrect results. When a
reproducible bug reveals a missing check, add the targeted check and a focused
regression test.

# Commits

Create commits only when explicitly requested by the user. Leave completed
work uncommitted until the user requests a commit.

Follow `.githooks/commit-msg` for the commit title format.
Write commit messages in English: one concise title line, a blank line,
and one or two body lines explaining the main changes and their purpose.

When a commit is requested, group related tasks into one coherent feature, fix,
or refactoring goal. Include the related implementation, tests, and
documentation together. Avoid separate commits for each small edit or
individual task. Split independent goals into separate commits; do not group
unrelated work just to reach a particular task count.

Before committing, review the staged diff to ensure every change belongs
to the commit's purpose and run the relevant checks. Each commit should
leave the project in a working state.
