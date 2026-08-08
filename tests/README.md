# Tests

**Owner:** PM & QA Lead

## Folder
```
tests/
```

## Files to Create

| File | Purpose |
|---|---|
| `test_module_a.py` | Unit tests for Perception Engine (YOLO detection, face recognition, item counting) |
| `test_module_b.py` | Unit tests for VLM Layer (VQA response format, TTS output, inference latency) |
| `test_module_c.py` | Unit tests for UI Dashboard (widget rendering, alert state transitions) |
| `test_backend.py` | Unit tests for Backend & Alerting (theft logic, Telegram payload, audit logging) |
| `test_integration.py` | End-to-end pipeline test: Camera → YOLO/Face Recognition → Logic → Alert → UI |
| `test_qa_scenarios.py` | Simulated incident scenarios: authorized user takes item vs. unauthorized intruder |

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
