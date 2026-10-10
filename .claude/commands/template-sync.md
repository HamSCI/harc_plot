# /template-sync: Adopt Changes from the Upstream ai_project_template

Use this when the session-start check reports that the upstream template has moved, or whenever
the user asks to bring the repository up to date with its template. The record of which template
this repository follows, and which commit of it was last adopted, is `.claude/template.json`.

The point of the record is that a template update is applied as a **change**: what the template
changed between the last-synced commit and its current tip is applied to this repository's
files. Copying the template's current files over this repository's would overwrite
project-specific content (names, funders, grant numbers, project rules), and must never be done.

## The record: `.claude/template.json`

In a project:

```json
{
  "role": "project",
  "upstream": "<owner>/ai_project_template",
  "upstream_branch": "main",
  "synced_commit": "<full SHA of the template commit last adopted>",
  "synced_on": "YYYY-MM-DD",
  "customized": ["CLAUDE.md", ".claude/rules/ai-governance.md", "ai/ai_usage_log.md"],
  "not_adopted": ["CITATION.cff", "docs/GETTING_STARTED.md"]
}
```

- `customized`: template files this repository has edited. They receive the template's change,
  never its whole file.
- `not_adopted`: template files this repository deliberately does not carry. Changes to them are
  reported and skipped.
- Every other template file present here is expected to match the template exactly.

In a template, `role` is `"template"`, `repo` names the template itself, and `siblings` maps each
sibling template to the sibling commit last reviewed here (see "In a template" below).

## Steps (in a project)

### 1. Fetch the template into this repository
```bash
up=$(python3 -c "import json;print(json.load(open('.claude/template.json'))['upstream'])")
br=$(python3 -c "import json;print(json.load(open('.claude/template.json')).get('upstream_branch','main'))")
git fetch --no-tags "https://github.com/$up.git" "+refs/heads/$br:refs/template/upstream"
old=$(python3 -c "import json;print(json.load(open('.claude/template.json'))['synced_commit'])")
git log --oneline "$old..refs/template/upstream"
git diff --stat "$old" refs/template/upstream
```
Fetching into a separate ref keeps the template's history out of this repository's branches,
and having its objects locally is what lets `git apply --3way` merge changes in step 3.

### 2. Explain the changes before touching anything
Summarize each upstream commit for the user in a sentence: what it changes and why (the commit
messages and the template's `ai/ai_usage_log.md` say). Skip `ai/ai_usage_log.md` itself: the
template's log belongs to the template.

### 3. Apply them, file by file
On a feature branch (`git switch -c template-sync-<date> origin/main`, or `master` where that is
the default), for each file the template changed:

- **`.claude/template.json` and `ai/ai_usage_log.md`:** never taken from the template. They
  are each repository's own record and log, and the session-start check ignores changes to them.
- **In `not_adopted`:** report it and skip it. A project's `README.md` is normally here: the
  template's README describes the template.
- **New in the template:** add it, unless it is clearly scaffolding this project has outgrown;
  ask if unsure, and record a declined file in `not_adopted`.
- **Not customized, and unchanged here since the last sync**
  (`git show "$old:<path>" | cmp -s - <path>` succeeds): take the template's new version,
  `git show refs/template/upstream:<path> > <path>`.
- **Customized, or edited here since the last sync:** apply only the template's change:
  ```bash
  git diff "$old" refs/template/upstream -- <path> | git apply --3way
  ```
  A conflict means the template changed text this project has rewritten. Resolve it by hand so
  that the project's content survives and the template's intent is carried over, and show the
  user that resolution explicitly. Never resolve by taking the template's side wholesale.
  Template placeholders (`{{...}}`) never overwrite filled-in project values.
- Where this repository already differs from the template on purpose (a `master` default branch,
  a committed-figures convention), keep the difference.

### 4. Record the sync
Set `synced_commit` to the full SHA of `refs/template/upstream`, `synced_on` to today's date,
and update `customized` and `not_adopted` if this sync changed them. Clear the day's cached
check so the next session re-checks: `rm -f "$(git rev-parse --absolute-git-dir)/template-check"`.

### 5. Commit through `/commit`
Log the session (purpose: adopting template changes `<old>..<new>`), commit with the
`[AI-assisted]` prefix, push the branch, and open a PR that lists each upstream change and how
it was applied. In a repository with submodules, each submodule that follows a template is synced
on its own, in its own PR.

## In a new project still carrying the template's record

A repository made with "Use this template" starts with the template's own record. The
session-start check notices that the repository is not the template and says so. Replace the
record with a project record: `upstream` is the template's `repo`, `synced_commit` is the
template commit the project was made from (the template's tip on the day, if not known
otherwise), `customized` lists the files placeholders were filled in, and `not_adopted` lists
template files the project deleted. Commit it with the instantiation.

## In a template

A template does not follow an upstream. It watches its **sibling templates**, because the
templates share workflow rules and a change made in one usually belongs in the others.

1. For each sibling in `siblings` whose tip has moved past `reviewed_commit`:
   ```bash
   git fetch --no-tags "https://github.com/<sibling>.git" "+refs/heads/main:refs/template/<name>"
   git log --oneline "<reviewed_commit>..refs/template/<name>"
   git diff "<reviewed_commit>" "refs/template/<name>" -- CLAUDE.md .claude README.md .gitignore
   ```
2. For each change, decide with the user whether it belongs in this template, and port it in
   this template's own words and context (the templates serve different audiences; do not copy
   blindly).
   A change this template already has (because it was made in all the templates at once) needs
   nothing more than marking reviewed.
3. Set that sibling's `reviewed_commit` to the full SHA reviewed, whether or not anything was
   ported, and commit through `/commit`, with a PR whose body links the sibling commits. A commit
   that changes only `.claude/template.json` does not make the siblings' checks fire in turn,
   because the check ignores changes to each repository's own record and log.

When you change workflow guidance in this template yourself, say in the PR body whether the
sibling templates need the same change, and open those PRs in the same session where they do.
