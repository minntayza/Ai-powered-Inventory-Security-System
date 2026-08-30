# Tests

**Owner:** PM & QA Lead

## Folder
```
tests/
```

## Test coverage

| File | Purpose |
|---|---|
Tests are split by public component instead of by team module. The suite includes perception/tracking, security policy, controller integration, camera recovery, incident recording, SQLite audit migration/retention, Telegram retry, siren/audio coordination, enrollment, dashboard rendering, VQA/TTS/voice input, and GPU health.

Run every test with:

```bash
python -m pytest -q
```

See `docs/QA_VALIDATION_RESULTS.md` for the automated evidence summary and the separate physical-hardware checklist.

## Also Responsible For

| Folder | Task |
|---|---|
| `docs/api_contracts/` | Define and maintain inter-module API schemas (YOLO output, VLM response, alert payload, event log) |
| `docs/presentations/` | Prepare pitch deck and slide files for exhibition |

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Run kickoff, establish Git repo, define API contracts, draft testing criteria |
| 2 | End-of-day module reviews, identify integration blockers, begin drafting docs + slides |
| 3 | Facilitate integration merge, run first end-to-end test (Camera → Detection → Alert) |
| 4 | Lead formal QA: simulate authorized vs. unauthorized incident, validate dashboard + alerts + VLM Q&A |
| 5 | Run multiple full-scale dry runs of exhibition pitch, finalize all deliverables |

## Test Scenarios to Cover

1. **Authorized user takes an item** — No alert should trigger
2. **Unknown person takes an item** — Theft alert + siren + Telegram notification
3. **Face briefly not visible** — Grace buffer prevents false alarm within 3 seconds
4. **VLM Q&A about incident** — Ask "What is the person doing?" and get a correct answer
5. **Network drop** — System recovers when camera reconnects

## Notes

- Tests should be runnable independently per module
- Integration tests require full hardware setup (camera + hotspot)
- Document all QA results for the final presentation
