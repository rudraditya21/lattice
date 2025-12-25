# Contributing to Lattice

## Code of Conduct
Please review `CODE_OF_CONDUCT.md`. By participating, you agree to uphold these standards.

## How to Contribute
- File issues with clear repro steps, expected vs actual behavior, and environment details.
- Before large changes, open an issue or RFC to align on scope/design (especially for language/IR changes).
- Keep PRs focused and small; prefer incremental steps with tests.

## Coding Style
- C++17, follow `.clang-format` and `.clang-tidy`.
- Prefer clear, explicit code over cleverness; avoid silent coercions.
- Deterministic behavior first: no hidden nondeterminism, seeded randomness only.
- Tests are required for new features and bug fixes (unit and, when applicable, regression tests).

## Testing
- Run `make test` (or `ctest --test-dir build`) before submitting.
- Add tests in `tests/` mirroring the area you change (lexer/parser/runtime/etc.).
- For parser/semantic changes, add coverage in docs when applicable.

## Reviews
- At least one maintainer review required.
- Address feedback with follow-up commits; avoid force-pushes unless asked.
- Document rationale in PR description; link related issues.

## Commit Hygiene
- Use descriptive commit messages (feature/bug scope).
- Keep commits logically grouped; avoid mixing formatting-only changes with logic changes.

## Security
- See `SECURITY.md` for reporting vulnerabilities.
