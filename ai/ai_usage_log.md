# AI Usage Log: harc_plot

This log records every substantive AI-assisted session on the project "HamSCI HARC database plotting toolkit".

Required by the HamSCI Generative AI Use Agreement, and by any institutional or funder policy
that applies to this project (see `.claude/rules/ai-governance.md`).

**This log is the source of truth for what AI did on this project.** A disclosure paragraph in a
manuscript, poster, or software release summarizes this log truthfully. Keep it current: an
entry written from memory weeks later is not a record.

## Entry format

```
## [YYYY-MM-DD HH:MM TZ]
- **Tool**: Claude (Anthropic), <exact-model-id>
- **Session Purpose**: What the session set out to accomplish
- **Sections/Files Affected**: Specific files, sections, or documents touched
- **Nature of Contribution**: Draft / Edit / Analysis / Code generation / Research / Scaffolding
- **Human Review Status**: Reviewed and verified / Partially reviewed / Pending review
- **Git Hash**: <filled in after committing>
```

The date and time come from the system clock via `date`, never from an estimate. The `/commit`
command produces this format and appends it in the right order.

---

<!-- Append new entries below this line, newest at the bottom. -->

## [2026-10-10 11:56 UTC]
- **Tool**: Claude (Anthropic), claude-opus-5-5 (1M context)
- **Session Purpose**: Add the HamSCI `ai_project_template` scaffolding (as of f7036e1) so work
  on this repository follows the branch-then-PR workflow and keeps its own AI usage log.
- **Sections/Files Affected**: `CLAUDE.md` (new, project-specific); `.claude/settings.json`,
  `.claude/commands/commit.md`, `.claude/rules/hamsci-data.md`, `.claude/rules/python-code.md`
  (template copies, with the default branch changed to `master`);
  `.claude/rules/ai-governance.md` (template, with the University of Scranton tier and the NASA
  80NSSC21K1772 / NSF AGS-2045755 funder tier filled in); `ai/ai_usage_log.md` (new);
  `.gitignore` (Claude Code local state).
- **Nature of Contribution**: Scaffolding.
- **Human Review Status**: Pending review.
- **Notes**: The grant numbers are those in the Madrigal HARC manuscript draft's acknowledgments,
  which carry a TODO to confirm them; the PI confirms the final list. CITATION.cff, issue
  templates and GETTING_STARTED.md from the template were not added.
- **Git Hash**: 5c97622, branch `template-pr-workflow`

## [2026-10-10 12:10 UTC]
- **Tool**: Claude (Anthropic), claude-opus-5-5 (1M context)
- **Session Purpose**: Make committing rendered output standing guidance, at NAF's direction
  (2026-10-10): memo and report PDFs and final figures capture the results of the code in
  human-readable form and must be committed, or past work is hard to look back on.
- **Sections/Files Affected**: `CLAUDE.md` ("Rendered Output"), `.claude/commands/commit.md` (step 2b), `.claude/rules/python-code.md`.
- **Nature of Contribution**: Edit (workflow guidance).
- **Human Review Status**: Pending review.
- **Git Hash**: 5631acc, branch `template-pr-workflow`

## [2026-10-10 12:35 UTC]
- **Tool**: Claude (Anthropic), claude-opus-5-5 (1M context)
- **Session Purpose**: Narrow the committed-output rule to curated documents, at NAF's direction
  (2026-10-10): "I'm looking more to capture PDFs that are memos, notes, or some more curated
  level of analysis", not large generated outputs such as figure batches.
- **Sections/Files Affected**: `CLAUDE.md` ("Curated Documents"), `.claude/commands/commit.md` (step 2b), `.claude/rules/python-code.md`.
- **Nature of Contribution**: Edit (workflow guidance).
- **Human Review Status**: Pending review.
- **Git Hash**: 30067ad, branch `curated-documents-only`
