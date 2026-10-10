#!/usr/bin/env bash
# Session-start check: has this repository's upstream ai_project_template moved on?
#
# Reads .claude/template.json, asks GitHub (public HTTPS, no credentials) for the tip of each
# template it tracks, and prints one line per template that has changed since it was last
# synced or reviewed. Claude Code adds that output to the session's context, so Claude can
# raise it with the user; /template-sync then does the work.
#
# Design limits, all deliberate:
#   - read-only: never changes a file;
#   - at most one network check per day per clone (cached in the git directory, untracked);
#   - silent on any failure (offline, no python3, no git, malformed record): it must never
#     slow down or break a session.
set -u

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
record="$root/.claude/template.json"
[ -f "$record" ] || exit 0
command -v python3 >/dev/null 2>&1 || exit 0
command -v git >/dev/null 2>&1 || exit 0

gitdir="$(git -C "$root" rev-parse --absolute-git-dir 2>/dev/null)" || exit 0
cache="$gitdir/template-check"
today="$(date -u +%Y-%m-%d)"

# Re-print today's result instead of asking GitHub again.
if [ -f "$cache" ] && [ "$(head -n 1 "$cache")" = "$today" ]; then
    tail -n +2 "$cache"
    exit 0
fi

msg="$(python3 - "$record" "$root" <<'PY' 2>/dev/null
import json, subprocess, sys

record_path, root = sys.argv[1], sys.argv[2]
rec = json.load(open(record_path))

def tip(repo, branch):
    """Commit SHA at the tip of a public GitHub branch, or None."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", f"https://github.com/{repo}.git", f"refs/heads/{branch}"],
            capture_output=True, text=True, timeout=8,
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"})
        return out.stdout.split()[0] if out.returncode == 0 and out.stdout else None
    except Exception:
        return None

# Files whose changes never need adopting: each repository's own record and AI log.
OWN = {".claude/template.json", "ai/ai_usage_log.md"}

def changed(repo, base, head):
    """Files changed between two commits (GitHub compare API), or None if unknown."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/compare/{base}...{head}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "template-check"})
        data = json.load(urllib.request.urlopen(req, timeout=8))
        return [f["filename"] for f in data.get("files", [])]
    except Exception:
        return None

def relevant(repo, base, head, ignore):
    """Changed files that matter here. Unknown file list: assume something does."""
    files = changed(repo, base, head)
    if files is None:
        return ["(file list unavailable)"]
    return [f for f in files if f not in ignore]

def summary(files):
    return ", ".join(files[:6]) + (f" and {len(files) - 6} more" if len(files) > 6 else "")

lines = []
role = rec.get("role", "project")

if role == "template":
    # A project freshly made from a template still carries the template's own record.
    try:
        origin = subprocess.run(["git", "-C", root, "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        origin = ""
    me = rec.get("repo", "")
    if origin and me and me.lower() not in origin.lower().replace(":", "/"):
        lines.append(f"[template-check] This repository still carries the template record of {me}. "
                     f"It looks like a new project made from that template: run /template-sync to "
                     f"record which template commit it started from.")
    else:
        # A template: watch the sibling templates for workflow changes worth porting.
        for repo, info in rec.get("siblings", {}).items():
            head = tip(repo, info.get("branch", "main"))
            seen = info.get("reviewed_commit", "")
            if head and seen and not head.startswith(seen):
                files = relevant(repo, seen, head, OWN)
                if files:
                    lines.append(f"[template-check] Sibling template {repo} has moved from {seen[:7]} "
                                 f"to {head[:7]}, changing {summary(files)}. Run /template-sync to "
                                 f"review those changes for ones this template should adopt.")
else:
    up = rec.get("upstream")
    if up:
        head = tip(up, rec.get("upstream_branch", "main"))
        seen = rec.get("synced_commit", "")
        if head and seen and not head.startswith(seen):
            files = relevant(up, seen, head, OWN | set(rec.get("not_adopted", [])))
            if files:
                lines.append(f"[template-check] Upstream template {up} has moved from {seen[:7]} "
                             f"(synced {rec.get('synced_on', '?')}) to {head[:7]}, changing "
                             f"{summary(files)}. Tell the user, and offer to run /template-sync "
                             f"to review and adopt the changes.")
print("\n".join(lines))
PY
)" || exit 0

printf '%s\n%s\n' "$today" "$msg" > "$cache" 2>/dev/null
[ -n "$msg" ] && printf '%s\n' "$msg"
exit 0
