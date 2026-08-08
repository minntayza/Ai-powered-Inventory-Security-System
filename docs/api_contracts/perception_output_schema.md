# Perception output contract

`PersonTracker.process_frame(frame)` returns a mapping containing an ISO timestamp,
frame number, detected people, and inventory state. Every person has `track_id`,
`bbox`, `name`, `authorized`, `authorization_state`, and confidence fields.

`authorization_state` is one of `authorized`, `unknown`, `not_visible`, or `spoof`.
Inventory includes raw/stable per-class counts, total counts, initialization state,
detected item boxes, and an optional persistent `change`/`drop` mapping.

