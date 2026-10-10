# /commit: AI-Assisted Commit Workflow

Use this command any time you finish a substantive AI-assisted work session.

This workflow handles flat repos (no submodules), single-submodule repos (for example Overleaf only), and multi-submodule repos. Submodules are auto-detected from `git submodule status`.

## Steps

### 1. Get the current timestamp
```bash
date
```
Use this exact output. Never estimate the date/time.

### 2. Identify all changes
Check status across the main repo and every submodule:
```bash
git status
git submodule foreach 'git status'
```

Show diff summaries:
```bash
git diff --stat
git submodule foreach 'git diff --stat'
```

If `git submodule foreach` produces no output, the repo has no submodules; proceed without them.

### 2a. Check what is about to be staged
Before drafting the log entry, look at what changed and confirm that none of it is material that
must not be committed (see `.claude/rules/hamsci-data.md`):

- Operator addresses, phone numbers, or personal email addresses
- Fine-grained station coordinates tied to an individual
- Unpublished data from a collaborator or a project instrument
- Credentials, API keys, tokens, or `.env` files with real values
- LaTeX intermediates (`.aux`, `.log`, `.bbl` and the like), large binary data, or identifiable
  student records. The compiled PDF is **not** an intermediate: it is committed with its source

If any callsigns or contributor names were added or changed in this session, verify them against
their source before committing. A misattributed observation is harder to retract than to prevent.

### 2b. Check the curated record is included
Every memo, note or report PDF this session produced or changed is staged alongside its source
and was rebuilt from the source being committed. Bulk generated output (figure batches, plot
sets, galleries, image compilations) is not staged; see `CLAUDE.md`.

### 3. Ask the user for session purpose
Draft a purpose of this session for the AI log. Show it to the user, and ask them to confirm.

### 4. Draft the AI usage log entry
Use this format. Use the **actual running model ID** in the Tool field (e.g., `claude-opus-5`), not a placeholder.

```
## [YYYY-MM-DD HH:MM TZ]
- **Tool**: Claude (Anthropic), <actual-model-id>
- **Session Purpose**: [user's description]
- **Sections/Files Affected**: [list changed files and sections]
- **Nature of Contribution**: [Draft / Edit / Analysis / Code generation / Research / etc.]
- **Human Review Status**: [Reviewed and verified / Partially reviewed / Pending review]
- **Git Hash**: [fill in after committing]
```

Present the draft to the user. Wait for confirmation or corrections.

### 5. Append the entry to `ai/ai_usage_log.md`

### 6. Every repo with changes: feature branch, commit, push the branch, open a PR
Nothing is committed directly to `master`, in this repo or any submodule. Work the submodules
first, then the main repo. For each repo with changes, propose a short kebab-case branch name
with the commit message and wait for the user's approval. Then:
```bash
git -C <repo> fetch
git -C <repo> switch -c <topic> origin/master     # uncommitted changes carry over to the branch
git -C <repo> add <files>
git -C <repo> commit -m "[AI-assisted] <description>"
git -C <repo> push -u origin <topic>
gh pr create --repo <owner>/<repo> --base master --head <topic> --title "..." --body-file <file>
```
If the repo is already on a feature branch whose PR is still open **for this same change**, add
the commit there and push it; do not open a second PR. If it is on some other branch, say so and
ask before branching.

The PR body says what changed and why, references the tracking issue, links the companion PRs
in the other repos of the same change, and ends with an AI attribution line. Pushing the feature
branch, opening the PR and pushing further commits to it are standing permission once the user
has approved the commit. **Never push `master`. Never merge**: a human maintainer reviews and merges. A repo
whose remote cannot host a PR (an Overleaf project, say) is outside this step; ask.

In the main repo, stage `ai/ai_usage_log.md` with the other changed files. **Do not stage a
submodule pointer that names an unmerged branch commit.**

A brand-new repository whose remote has no `master` yet is the one exception: its first commit
seeds `master`. Ask before that push.

### 7. Fill in the git hash(es)
```bash
git log --oneline -1
```
Update the entry's **Git Hash** field with the branch commit and PR for each repo (e.g.
`main-repo=abc1234 (PR owner/repo#5), sub=def5678 (PR owner/sub#3)`). Commit that on the same
branch with a plain message without the `[AI-assisted]` prefix, such as
`Update AI usage log with git hashes`, and push it.

That is the last log update for this change, and it never needs a PR of its own. A merge commit
keeps every branch SHA, so the recorded hash stays valid after the merge. If a PR is squash- or
rebase-merged, its branch SHAs no longer exist: correct them to the merged SHA on the next
branch that touches the log.

### 8. After a submodule PR merges: bump the pointer
```bash
git -C <sub> switch master && git -C <sub> pull --ff-only && git -C <sub> branch -d <topic>
git add <sub> && git commit -m "Bump <sub> to merged PR #N"
```
Make that commit on the main repo's feature branch: the still-open PR for the same change if
there is one, otherwise a new branch and PR. The pointer-bump commit is this repo's record of
the merged submodule SHA; the log needs no further update.

### 9. Pushing
Never push `master`. The feature-branch pushes above are the one standing exception; ask before
any other push. Fetch and verify the remote state first, push fast-forward only, never
force-push, never hard-reset.

With submodules, **push the submodule before the parent**. A parent that reaches GitHub ahead
of its submodule looks correct on the machine that pushed it and breaks for everyone who clones.
Verify first:
```bash
git -C <submodule-path> branch -r --contains HEAD   # empty output = local only; push it first
```

## Notes

- The `[AI-assisted]` prefix applies only to commits whose content was produced or substantially shaped with AI assistance. Pure human edits (the user manually fixes a typo or rewords a sentence) should NOT carry the prefix.
- Commit submodules with their own `[AI-assisted]` prefix when their content is AI-assisted, separately from the main-repo pointer-bump commit.
- Use `git add <specific-files>` rather than `git add -A` or `git add .`, to avoid accidentally staging untracked artifacts (build outputs, credentials, large binaries).
- Write the commit message for someone reading it in two years with no context: what changed and why, not a restatement of the diff.
- Reference the tracking issue (`refs #N`, or `org/repo#N` when the issue lives in another repo). Use `closes #N` only when completion is yours to declare; it fires the moment the commit reaches the default branch, which silently overrides anyone who was asked to verify the fix and close it themselves.
- More than one session may be working the same repository at once. Run `git fetch && git status -sb` before starting a pass that will touch many files, not only before pushing.
