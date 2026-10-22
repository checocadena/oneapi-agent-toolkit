# Meta Global AI Developer Hackathon: working notes

Entry: oneapi-agent-toolkit, extended so Meta's Muse models drive the porting workflow.
These notes are updated as decisions are made. **Event facts below are from the event page
and must be re-checked against the Official Rules (published Oct 26).**

| File | What |
|---|---|
| [`AUDIT.md`](AUDIT.md) | State of the repo before the hackathon: what exists, what is missing |
| [`FEATURE-OPTIONS.md`](FEATURE-OPTIONS.md) | Candidate Meta-powered features, with scope and cost |
| [`SUBMISSION-OUTLINE.md`](SUBMISSION-OUTLINE.md) | Placeholder submission: summary, Meta usage, demo script |
| [`../meta-model-api.md`](../meta-model-api.md) | How the loop talks to the Meta Model API; day-one checklist |

## Event facts (to verify against the Official Rules)

- Hosts: Meta and Cerebral Valley; fully remote. https://cerebralvalley.ai/e/meta-ai-developer-hackathon
- Oct 26: kickoff (about 1 pm ET) and Meta Model API access.
- Oct 27 to Nov 4: workshops and Discord office hours.
- **Nov 4, 9:00 pm ET: submissions due.** Target: submit by midday Nov 4.
- Nov 4 to 8 judging; Nov 9 closing ceremony.
- $150 of Meta Model API credit per participant.
- Tracks: indie developer/student, or startup/team. **Decision pending** (see below).
- Models named on the event page: Muse Spark series, Muse Image, SAM 3.1 (Meta Model API),
  Meta open-weights models, Muse Code.

### Open questions for the Official Rules

- [ ] Are pre-existing projects allowed? If yes, how must the existing work be disclosed?
- [ ] Must the entry use Meta models, and how centrally?
- [ ] Submission fields, video length, repository requirements.
- [ ] Licensing or IP terms (any conflict with AGPL-3.0?).
- [ ] Which track fits a solo, independent developer.
- [ ] Rules on using other vendors' models (the existing Claude results) in the same project.

## Gates

| Date | Gate | Status |
|---|---|---|
| by Oct 25 | Repo audited; provider interface + mock client working; README cleaned up | ✅ done on branch `meta-hackathon-prep` (needs review and merge) |
| Oct 26 | Real Meta Model API swapped in (day-one checklist); Official Rules read; this file updated | ⏳ |
| Oct 29 to Nov 2 | Build the chosen Meta feature | ⏳ choose from FEATURE-OPTIONS.md |
| Nov 3 | Record the demo video; finalize the submission | ⏳ |
| Nov 4 | Submit by midday | ⏳ |

## Working rules

- Original work only; third-party components and licenses are listed in the README.
- No secrets in the repository: keys come from environment variables (`.env.example`).
- No Meta-model result is claimed until it is measured and its evidence is committed.
- Every paid step (API calls, GPU rentals) gets a cost estimate and an explicit OK first,
  with a hard cap per run and per batch.
- Small commits; larger refactors or public-API changes are proposed before they are made.
- Decisions go in [`DECISIONS.md`](../../DECISIONS.md), progress in [`PROGRESS.md`](../../PROGRESS.md).
