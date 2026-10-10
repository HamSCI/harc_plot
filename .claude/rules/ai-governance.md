# AI Governance and Policy Compliance

All AI-assisted work on this project must comply with the policies below. HamSCI is a
public-facing collaboration between professional researchers, students, and amateur radio
operators. Undisclosed or careless AI use risks the project's credibility with funders, with
the scientific community, and with the amateur radio community whose volunteered observations
make the science possible.

The policy stack is **tiered**. Tier 1 binds every HamSCI project. Tier 2 (University of Scranton)
and Tier 3 (NASA and NSF) both apply to this project.

---

## Tier 1: Always binding on every HamSCI project

### HamSCI Generative AI Use Agreement (January 2026)
- Log the AI tool name, version, and date/time for every substantive session
- Verify all AI-generated output before it enters a project artifact
- Do not submit confidential, ITAR/EAR-controlled, or proprietary material to AI tools

### Universal rules

**Never fabricate.** No invented citations, references, data, quotations, numbers, or
attributions. If a number cannot be traced to a script, a dataset, or a paper that was actually
opened, it does not go in the text. This applies with particular force to callsigns, station
identifiers, and observer names: an invented callsign attributes work to a real operator who
did not do it.

**Humans own the science.** Scientific claims, figure interpretations, methodology choices, and
conclusions belong to the human authors. A human reviews all AI-generated content before it
enters an artifact. Verify factual claims against authoritative sources before publication.

**AI is not an author.** AI tools cannot be listed as authors, contributors, or co-investigators.
They cannot take responsibility for the work, assert the presence or absence of conflicts of
interest, or manage copyright and license agreements. Keep AI tools out of author blocks,
acknowledgment-as-contributor lines, CRediT statements, and `CITATION.cff` author lists.

**Never submit restricted material to an AI tool.** See `.claude/rules/hamsci-data.md` for what
counts as restricted in a HamSCI context, including the handling of operator personal data.

**Log before you commit.** Every substantive AI session is recorded in `ai/ai_usage_log.md`
before the work is committed. See the logging requirements below.

**Disclose AI use in published outputs.** Any manuscript, poster, presentation, or software
release whose content was produced or substantially shaped with AI assistance says so, naming
the tool and describing what it did. AGU's author policy is the reference standard:

> "Authors who use AI tools in the writing of a manuscript, production of images or graphical
> elements of the paper, or in the collection and analysis of data, must be transparent by
> disclosing details of use, including which AI tool was used and how it was used, in the
> Materials and Methods (or similar section) of the paper. Authors are fully responsible for
> the content of their manuscript, even those parts produced by an AI tool."

Check the specific venue's policy, which may differ. Keep the disclosure short: name the tool
and version, state concretely what it was used for, state what the authors did, and stop. The
disclosure must stay accurate as scope changes; understating scope is the failure mode to avoid.

---

## Tier 2: Institutional policy

### University of Scranton AI Policy (September 2025)
- Maintain transparency about AI use in all project outputs
- Human oversight and review of all AI-generated content is required
- Do not use AI to misrepresent authorship or intellectual contribution
- Ethical use aligned with the University's academic integrity standards

Contributors at other institutions work under their own institutional policies. **Students**:
your institution's academic integrity policy applies to coursework-adjacent research, and it may
be stricter than anything here.

---

## Tier 3: Funder policy

### NASA Guidance on Generative AI in Funded Research
- Disclose AI-assisted content in deliverables to NASA-funded projects
- Maintain human authorship and accountability for scientific claims
- Do not submit ITAR/EAR-controlled, confidential, or unpublished mission data to AI tools
- Verify factual claims against authoritative sources before publication

### NSF Guidance on Responsible Use of Generative AI in Funded Research
- Disclose AI use in NSF deliverables, proposals, and publications as required
- Maintain human authorship and intellectual responsibility for results
- Do not use AI to generate or substantially shape proposal review content unless explicitly
  authorized
- Do not submit confidential or unpublished data to AI tools

### Funding for this repository
The HamSCI amateur radio datasets in Madrigal, and the code that analyzes them, were developed
under NASA grant 80NSSC21K1772 and NSF grant AGS-2045755, the grants currently acknowledged in
the Madrigal HARC 2026 manuscript draft. The PI confirms the final grant list for each published
output; list every funder's grant number in its acknowledgments.

AI-assisted content that materially affects a collaborator-led section should be flagged to that
collaborator, and is subject to **their** institutional and funder policies as well as ours.

---

## AI Usage Logging Requirements

**Every substantive AI session is logged in `ai/ai_usage_log.md` before the work is committed.**

Each entry must include:

| Field | Requirement |
|---|---|
| **Date/Time** | From the system clock via `date`. **Never estimated.** |
| **Tool** | Name and exact version, e.g. "Claude (Anthropic), claude-opus-5" |
| **Session Purpose** | What the session set out to accomplish |
| **Sections/Files Affected** | Specific files, sections, or documents touched |
| **Nature of Contribution** | Draft, edit, analysis, code generation, research, scaffolding |
| **Human Review Status** | Reviewed and verified / Partially reviewed / Pending review |
| **Git Hash** | Added after committing |

Work done inside a submodule is logged in that submodule's own log where one exists. The
internal log is the **source of truth** for what AI did; a manuscript or release disclosure
summarizes it truthfully.

Use the `/commit` command, which handles logging and committing in the correct order.

### Marking AI-drafted issues and comments

`Co-Authored-By:` is a commit-message trailer. GitHub parses it only for commits, so issues,
issue comments, and pull request descriptions are attributed solely to the authenticating
account. AI-drafted content on those surfaces is invisible unless it is marked by hand. Mark it:

```html
<sub>Drafted by Claude (Anthropic), `claude-opus-5`, under N. A. Frissell's direction; analysis
verified against the sources named above. Scientific decisions are the authors'.<br>
Co-Authored-By: Claude Opus 5 &lt;noreply@anthropic.com&gt;</sub>
```

Keeping the `Co-Authored-By:` string identical to the one used in commits means
`gh issue list --search "Co-Authored-By"` finds AI-drafted issue content the same way
`git log --grep` finds AI-assisted commits. This is disclosure, not authorship; it never
appears in a manuscript, an acknowledgment, or a `CITATION.cff`.

---

## What AI Should NOT Do

- Fabricate citations, references, data, quotations, callsigns, or observer attributions
- Claim authorship, or present AI output as purely human work
- Submit restricted material to an AI tool (see `.claude/rules/hamsci-data.md`)
- Skip the AI usage log before committing AI-assisted changes
- Commit or push directly to `master`, or merge a pull request
- Push anything other than an approved feature branch without explicit instruction
- Force-push or hard-reset without explicit instruction
- Modify human-curated surfaces (project boards, issue status, milestones) without instruction
