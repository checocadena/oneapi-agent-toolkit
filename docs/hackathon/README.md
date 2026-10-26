# Meta Global AI Developer Hackathon: working notes

Entry: oneapi-agent-toolkit, extended so Meta's Muse models drive the porting workflow.
These notes are updated as decisions are made. Event facts come from the Participant Guide.

| File | What |
|---|---|
| [`AUDIT.md`](AUDIT.md) | State of the repo before the hackathon: what exists, what is missing |
| [`FEATURE-OPTIONS.md`](FEATURE-OPTIONS.md) | Candidate Meta-powered features, with scope and cost |
| [`SUBMISSION-OUTLINE.md`](SUBMISSION-OUTLINE.md) | Placeholder submission: summary, Meta usage, demo script |
| [`../meta-model-api.md`](../meta-model-api.md) | How the loop talks to the Meta Model API; day-one checklist |

## From the Participant Guide (read 2026-10-26)

- **Building period: Oct 27 to Nov 4.** Submissions due **Wed Nov 4, 9:00 pm ET** on the
  Cerebral Valley platform. Judging Nov 5 to 8; top 5 per track and winners Nov 9.
- **Problem statement:** "Push the Frontier" (software is named explicitly).
- **Must use Meta's models** (Meta Model API, Meta open-weight models, or Muse Code).
- **Repository must be public.** Solo entries allowed.
- **New work only.** What is presented must be built during the event, and the demo may only
  show features built during the event; failing to identify new contributions clearly means
  disqualification. *Startups* may build on existing codebases if the new parts are clearly
  identified. *(Whether an indie entry may extend its own existing open-source project the
  same way is not stated: asked in Discord #questions.)*
- **Submission:** a 3-minute demo video of the new features, the code, and the live product.
- **Judging:** Stage 1 pass/fail (fits a problem statement, real use of Meta models); then
  Impact 30%, Use of Meta AI tooling 25% ("beyond a basic integration"), Technical depth 20%,
  Demo 25%.
- **Bonus prizes that fit:** Best Agentic Use-Case ($5k), Best Build on Meta Open Weight
  Models ($10k: Muse Spark Open or Muse Glimmer, own hardware or a partner such as Baseten
  or Modal), Most Viral on X ($20k), Community Vote ($10k).
- **Credits:** $150 Meta Model API (claim form; a card is required; set a spend limit on the
  Billing page); Baseten $100 and Modal $100 (by request); Fireworks $25 (shortlist).
- **Sessions worth attending:** Oct 28 "Agents 102: Best Practices for Verifiable Agents";
  Oct 30 Fireworks on Muse open models.

### Open questions

- [ ] Track: Indie Developer (solo) or Startups. Startups may explicitly build on existing
      code; for indie entries, extending one's own open-source project is not addressed.
- [ ] Muse model ids, base URL and pricing (day-one checklist in `../meta-model-api.md`).

## Pre-existing work and the hackathon boundary

Everything on `main` up to the tag `pre-hackathon` (set before Oct 27) existed before the
event: the toolkit, its Claude measurements, the GPU pipeline, and the provider layer
(PR #18). Hackathon work starts after that tag, in commits dated Oct 27 or later, and is
listed in the submission. The demo shows only that new work; the toolkit appears as the
existing foundation it runs on.

## Gates

| Date | Gate | Status |
|---|---|---|
| by Oct 25 | Repo audited; provider interface + mock client working; README cleaned up | ✅ done on branch `meta-hackathon-prep` (needs review and merge) |
| Oct 26 | Official Rules read; this file updated | ✅ |
| Oct 26 to 27 | Real Meta Model API swapped in (day-one checklist) | ⏳ needs the key |
| Oct 26 | Merge PR #17 and #18, tag `pre-hackathon` | ⏳ |
| Oct 27 to 28 | Muse comparison runs (supporting evidence) and the Muse smoke test | ⏳ needs the spending cap |
| Oct 29 to Nov 2 | Build the hackathon feature: Muse Port (planner + verified port of llm.c kernels) | ⏳ |
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
