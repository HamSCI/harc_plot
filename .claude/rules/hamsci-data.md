# HamSCI Data, Callsigns, and Volunteer Attribution

Applies to every HamSCI project. HamSCI science runs on observations contributed by amateur
radio operators, most of whom are volunteers rather than employees or students. Their callsigns
and station locations are simultaneously **scientific metadata**, **personal credit**, and, in
some cases, **personally identifying information**. Handle all three at once.

---

## Callsigns and operator names

**A callsign identifies a specific licensed human being.** Treat it as you would treat a named
author, because in practice it is one.

- **Never invent a callsign.** A fabricated callsign is not a harmless placeholder. It is a real,
  or potentially issuable, license, and using one attributes an observation to an operator who did
  not make it. Where documentation genuinely needs an example callsign, use the conventional
  placeholder `N0CALL`, or a callsign belonging to a project member who agreed to it, and say in
  the surrounding text that it is an example.
- **Never invent an association** between a callsign and a station, a measurement, a location, or
  a person. If the link between a callsign and an observation cannot be traced to a dataset or a
  registration record that was actually read, it does not go in the text.
- **Spell callsigns exactly as licensed**, including portable and secondary-station suffixes
  (`W2NAF`, `KD8OXT`, `VE3xxx/W2`). A mangled callsign misattributes the work.
- **Verify before publishing.** Callsigns are checkable against public license databases. Check
  them rather than trusting a transcription, and check that the operator's name matches the
  record when you name them.

---

## Station location, grid squares, and personal data

An amateur station is very often someone's home. A station coordinate at full precision is very
often a residential address.

- **Publish location at the coarsest precision the science supports.** A 4-character grid square
  (roughly 1° × 2°) or a 6-character grid square (roughly 2.5' × 5') is adequate for most
  ionospheric work. Publishing 8-character grid squares or decimal-degree coordinates to five
  places identifies a house.
- **Amateur license records are public in some jurisdictions and not in others.** In the United
  States, the FCC license database is public and has historically included a licensee mailing
  address. Many other national authorities publish the callsign without the address. Do not
  assume that because an operator's address is discoverable somewhere, it is appropriate to
  republish it in a paper, a figure, a repository, or a dataset.
- **Do not put operator addresses, phone numbers, or email addresses in a tracked file**, and do
  not put them in an AI prompt. Station metadata that a network distributes (callsign, grid
  square, hardware, antenna) is the part meant to travel with the science.
- **Honor opt-outs.** If a contributor asks not to be named, or asks that their location be
  coarsened, that request governs, in every artifact including the ones already drafted.

---

## Community-contributed datasets

HamSCI analyses frequently use data from networks the project does not itself run: PSWS and
Grape receivers, WSPRNet, the Reverse Beacon Network, PSKReporter, contest and QSO party logs,
and SuperDARN, Madrigal, and GNSS products from the professional side.

- **Read the relevant network's own terms before redistributing its data.** Each network sets its
  own conditions on reuse, attribution, and bulk access; they are not interchangeable, and they
  are not inferable from the fact that the data is downloadable.
- **Cite the network and the data**, with its access date, the same way you would cite a paper.
  Cite the software that produced the measurement, and not only the software written for this
  study.
- **Do not upload bulk community data to an external AI tool.** Analyzing it locally, in a
  governed project environment, is the intended workflow. Pasting a dataset into a public chat
  session is what these rules prohibit.
- **Unpublished data stays unpublished.** Data from an instrument the project runs, or shared by
  a collaborator ahead of their own publication, does not go to an external AI tool and does not
  go into a public repository before the owner has released it.

---

## Crediting volunteers

Volunteer observers are the reason the dataset exists, and they read the papers.

- **Credit contributors by name and callsign** where they consented to be named, in the
  acknowledgments, in the author list where the contribution warrants authorship, or in a
  contributor table in the paper or its data product.
- **Get the roles right.** Do not describe a volunteer as a student, an employee, or an
  investigator, and do not describe a student's contribution as a volunteer's. Ask rather than
  guess.
- **Describe actual contributions**, not aspirational or approximate ones. An AI-drafted
  acknowledgment that pads a contributor list, or that assigns a role nobody verified, is a
  fabrication with a real person's name attached to it.
- **Authorship decisions belong to the humans on the project**, and follow the venue's authorship
  criteria. AI does not propose author lists.

---

## Quick reference: what may go to an external AI tool

| Category | Status |
|---|---|
| Published data from a public network, within its terms | **OK** |
| Code the project owns | **OK** |
| Draft prose in a private, governed project repository | **OK** |
| Callsigns as they appear in a public registration or dataset | **OK**, in project context |
| Operator home addresses, phone numbers, personal email | **Never** |
| Fine-grained station coordinates tied to an individual | **Never** |
| Unpublished data from a collaborator or a project instrument | **Never** |
| ITAR/EAR-controlled, confidential, or proprietary material | **Never** |
| Student grades, evaluations, or identifiable academic records | **Never** |
| Anything posted to a **public** AI chat, forum, or Q&A site | **Never**, for all of the above |

The distinction that matters: working with sensitive material inside a governed, private project
environment is the intended workflow. Redistributing it outward is what is prohibited.
