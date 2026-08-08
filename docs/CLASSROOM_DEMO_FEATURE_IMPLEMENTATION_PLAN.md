# Classroom Demo Feature Implementation Plan

## Objective

Turn the perception prototype into a controlled, repeatable classroom security
demonstration without requiring custom YOLO training. Pretrained COCO classes
remain visible, while only explicitly protected objects can create an alert.

## Phase 1: inventory policy, zone, and baseline

- Split COCO classes into `protected` and `contextual` policy lists.
- Reject overlapping, unknown, or `person` inventory policy entries at startup.
- Count protected detections only when their center is inside the monitored zone.
- Keep contextual detections visible without allowing them to trigger theft.
- Start `DISARMED`; require **Set baseline and arm** after counts stabilize.
- Keep the baseline fixed until reset and do not persist it across restarts.
- Support `DISARMED`, `ARMED`, and `PAUSED` independently of security state.
- Allow resume only when protected counts still match the baseline.
- Persist the normalized zone in ignored runtime storage.
- Disarm and clear the baseline whenever the zone or frame source changes.

Acceptance: contextual movement cannot alert, no event confirms while disarmed,
and one stable protected decrease produces one discrepancy.

## Phase 2: attribution and incident evidence

- Preserve ByteTrack inventory IDs and use same-label IoU fallback IDs.
- Retain two seconds of item/person proximity evidence.
- Score actors using expanded-person-box overlap and normalized distance.
- Mark close candidate scores ambiguous instead of choosing arbitrarily.
- Attach candidates as evidence; attribution alone never triggers theft.
- Migrate SQLite in place with actor, baseline, decision, zone, source, and video fields.
- Buffer ten seconds before confirmation and record ten seconds afterward.
- Write asynchronously as MP4, with an MJPEG AVI fallback.
- Preserve the event and snapshot if video encoding fails.
- Remove expired media according to `retention_days` while retaining audit rows.

Acceptance: confirmed incidents include deterministic reasoning and available
evidence, buffering stays bounded, and existing databases migrate without loss.

## Phase 3: safe replay mode

- Accept MP4, AVI, and MOV uploads through the dashboard.
- Replay at source speed capped by configured processing FPS, with optional looping.
- Reset baseline, tracker history, evidence buffer, and security state on switching.
- Mark replay events with `source_type: replay`.
- Always suppress siren and Telegram in replay mode.
- Clean temporary uploads using the configured retention policy.

Acceptance: recorded demonstrations are repeatable, cannot send external alerts,
and do not retain stale identities or tracking IDs.

## Phase 4: enrollment and visual investigation

- Add atomic enrollment for one to five JPG/PNG images.
- Validate size, extension, exactly one face, and minimum face dimensions.
- Permit safe identity names and use UUID image filenames.
- Refresh recognition immediately after enrollment or removal.
- Require explicit confirmation before deleting an identity.
- Ignore new biometric images in Git and warn about OneDrive synchronization.
- Add **Describe scene**, **Read text**, and **Summarize incident** actions.
- Route descriptions to Florence detailed captioning and text requests to OCR.
- Combine structured evidence with captions for incident summaries.
- Keep all VLM output advisory and outside the security state machine.

Acceptance: enrollment works without a restart, invalid batches are atomic, and
VLM failure never interrupts monitoring.

## Phase 5: performance and hardening

- Retain 60 in-memory samples for FPS and component latency.
- Display YOLO/face/VLM time, processing average/maximum, skipped frames,
  evidence-buffer memory, and the active CUDA/MPS/CPU device.
- Reset source-specific metrics when switching between live and replay.
- Keep telemetry memory-only and preserve accelerator fallback behavior.
- Cover every new policy, state transition, persistence path, and validator in tests.

## Runtime contracts

### Monitoring snapshot

```python
{
    "mode": "DISARMED | ARMED | PAUSED",
    "baseline_ready": True,
    "baseline_counts": {"laptop": 1},
    "current_counts": {"laptop": 1},
    "missing_items": {},
    "extra_items": {},
    "shelf_region": [0.1, 0.2, 0.9, 0.8],
}
```

### Inventory item

```python
{
    "label": "laptop",
    "bbox": [100, 120, 300, 310],
    "confidence": 0.91,
    "track_id": 12,
    "policy": "protected",
    "in_zone": True,
}
```

### Security-event additions

```python
{
    "baseline_counts": {"laptop": 1},
    "primary_actor": {"track_id": 4, "name": "Unknown", "association_score": 0.73},
    "actor_candidates": [],
    "decision_reason": "protected_item_removed_with_unauthorized_or_unknown_person",
    "zone_region": [0.1, 0.2, 0.9, 0.8],
    "source_type": "live",
    "video_path": "data/logs/incidents/<event-id>.mp4",
}
```

## Test matrix

- Policy validation: overlaps, missing targets, and `person` rejection.
- Counter: noise, fixed baseline, one-shot discrepancies, recovery, and resume.
- Zone: protected items outside the region and people intersecting it.
- Attribution: clear, ambiguous, missing candidate, and fallback item IDs.
- Security: authorized, unknown, mixed, absent person, recovery, and cooldown.
- Persistence: old database migration, JSON fields, media paths, and retention.
- Replay: formats, looping, state reset, and external-alert suppression.
- Enrollment: valid batch, invalid batch, unsafe name, traversal, and deletion.
- Performance: bounded samples, FPS, frame gaps, and source reset.
- Compatibility: CPU, CUDA, Apple MPS, dashboard import, and VLM task routing.

## Classroom workflow

1. Start Streamlit and wait for Camera and Models to become ready.
2. Adjust the monitored rectangle around the demonstration desk or shelf.
3. Place protected items inside it and wait for stable counts.
4. Click **Set baseline and arm**.
5. Move a contextual object to show that it cannot trigger security.
6. Perform an authorized removal and show the cancelled audit event.
7. Perform an unknown-person removal and wait for confirmation.
8. Acknowledge the alert and inspect actor evidence, snapshot, and video.
9. Use safe replay mode when a live scenario is unreliable.
10. Use Florence only after selecting a live frame or stored event image.

## Defaults and constraints

- One camera and one monitored rectangle are supported.
- Baselines are runtime-only; the zone persists locally.
- Default classes use pretrained COCO labels, so custom training is optional.
- Custom training is needed only for objects COCO cannot recognize reliably.
- SQLite and runtime media stay under `data/logs/`.
- Replay never activates siren or Telegram.
- Face images stay on the workstation but may synchronize through OneDrive.
- Florence never authorizes people or alters alert decisions.
