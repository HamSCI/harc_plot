# harc_plot: HamSCI Amateur Radio Communications Database Plotting Toolkit

## Project Overview
Python toolkit for loading, binning and plotting HF amateur radio propagation spot data from the
Reverse Beacon Network, WSPRNet and PSKReporter, as distributed through the MIT Haystack CEDAR
Madrigal database (`rsdYYYY-MM-DD.01.hdf5`), with legacy CSV support. It also bundles F10.7 from
NASA OMNIWeb and retrieves GOES XRS X-ray irradiance from NOAA NCEI. Public, GPL-3.0; used by the
Madrigal HARC 2026 paper, the Sanchez LSTID climatology and other HamSCI analyses.

**Lead**: Nathaniel A. Frissell, W2NAF, University of Scranton
**Collaborators**: see the commit history and README acknowledgements
**HamSCI Working Group**: none
**Institution**: University of Scranton
**Funder**: NASA (80NSSC21K1772) and NSF (AGS-2045755), per the current Madrigal HARC manuscript
draft
**Project period**: ongoing

## Project Goal
A dependable, documented library that turns Madrigal HARC files into histograms and figures,
with the data-quality problems of the source files (for example the per-file `ut1_unix` offset)
handled in code and documented where they are fixed.

## Standing Rules

These two are binding on every HamSCI project and are imported here so they load into context
automatically:

@.claude/rules/ai-governance.md
@.claude/rules/hamsci-data.md

`.claude/rules/python-code.md` applies to `.py`, `pyproject.toml`, and `requirements*.txt`
through its own `paths:` frontmatter.

## Repository Structure

```
harc_plot/
├── CLAUDE.md                     ← this file; project instructions for Claude
├── README.md                     ← install, data input, data sources and acknowledgements
├── LICENSE                       ← GPL-3.0
├── setup.py, requirements.txt
├── .claude/                      ← settings, /commit, rules
├── ai/ai_usage_log.md            ← mandatory AI session log
├── harc_plot/                    ← the package
├── scripts/                      ← drivers, one directory per task
├── data/                         ← small bundled inputs (sample Madrigal file, OMNI F10.7)
├── output/                       ← committed reference output
└── example_plots/
```

## Working Conventions

**The default branch is `master`, not `main`.** Everything below that says "the default branch"
means `master`.

**Commits.** Use the `/commit` command. It logs the AI session, then commits on a feature branch
and opens a pull request. Prefix AI-assisted commits with `[AI-assisted]`. Reference tracking
issues (`refs #N`, or `closes #N` only when completion is yours to declare).

**Every change goes through a pull request.** Branch from `origin/master`, commit on the branch,
push the branch, and open a PR with `gh pr create --base master`. A human maintainer reviews and
merges. Claude never commits or pushes directly to `master` and never merges.

**Pushing a feature branch and opening its PR is standing permission** once the user has approved
the commit. Any other push needs explicit instruction. Never force-push or hard-reset. Fetch and
verify remote state before any push.

**Downstream repositories pin this one as a submodule.** A parent may only point at a commit that
has reached a remote, and under the PR workflow, at a merged one.

**Public repository.** Acknowledgement wording for bundled third-party data follows each
provider's own citation guidance; keep it in the README beside the data it covers.

## AI Governance

Every substantive AI session is logged in `ai/ai_usage_log.md` **before** the work is committed.
Use `/commit`, which enforces the ordering. The full policy stack is in
`.claude/rules/ai-governance.md`, which is imported above.

AI work on this repository before 2026-10-10 (commits prefixed `[AI-assisted]`) is logged in the
AI usage log of the repository it was done for, chiefly
`w2naf-academia/madrigal_harc_2026_support`.
