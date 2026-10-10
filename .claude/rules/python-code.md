---
paths:
  - "**/*.py"
  - "pyproject.toml"
  - "requirements.txt"
  - "requirements*.txt"
---

# Python Code Rules
*Delete this file if your project does not include Python code.*

## Code Standards
- Follow existing code structure; do not introduce new abstractions without a reason you can state
- Maintain a `requirements.txt` (or `pyproject.toml`) and keep it up to date when adding dependencies
- Do not commit credentials, API keys, or sensitive configuration. Use environment variables or a `.env` file that is gitignored

## Commit Workflow for Code
- If code lives in a submodule, commit changes in the submodule first, then update the pointer in the main repo
- Use `[AI-assisted]` prefix on AI-assisted commits
- Reference issue trackers (GitHub issues, project boards) in commit messages where applicable (e.g., `closes #N`)
- Every change goes on a feature branch and through a pull request; never commit or push directly to `master`, and never merge (a human maintainer merges)
- Pushing a feature branch and opening its PR is standing permission once the user approves the commit; ask before any other push

## Data Handling
- Commit the rendered results a script produces (final figures, report PDFs) with the script, so the
  repository shows what the code found; bulk inputs, caches and scratch output stay out
- Do not commit bulk or raw data. Fetch it from its archive with a committed script, or regenerate it with a committed builder, so anyone can reproduce your inputs
- Do not hard-code operator personal data, fine-grained station coordinates, or credentials into a script. See `.claude/rules/hamsci-data.md`
- Cite the software your results depend on, including the upstream library that produces the primary measurement, and not only the code written for this project

## Open Source Considerations
- If the code will be released open-source, keep commit history clean and suitable for public visibility
- Document significant design decisions in code comments or in a `docs/` directory
- Apply the project's chosen open-source license consistently (`LICENSE` file at the repo root)
