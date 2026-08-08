# Perception output contract

`PersonTracker.process_frame(frame)` returns a mapping containing an ISO timestamp,
frame number, detected people, and inventory state. Every person has `track_id`,
`bbox`, `name`, `authorized`, `authorization_state`, and confidence fields.

`authorization_state` is one of `authorized`, `unknown`, `not_visible`, or `spoof`.
Inventory includes protected/contextual stable counts, the fixed protected
baseline, missing/extra items, detected boxes, and optional `change`/`drop`
mappings. Each item includes `track_id`, `policy`, and `in_zone`. Only protected
items inside the monitored zone participate in the baseline.

The controller snapshot publishes `monitoring` with mode, baseline readiness,
baseline/current counts, discrepancies, and the normalized shelf region.
Monitoring mode is independent from the theft-detector state.
