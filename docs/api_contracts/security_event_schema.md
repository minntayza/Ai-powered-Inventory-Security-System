# Security event contract

Events have a UUID `event_id`, ISO timestamp, `event_type`, status, before/after
counts, removed item quantities, associated track IDs, observed authorization
states, optional snapshot path, Telegram delivery status, and acknowledgement flag.

Events also include baseline counts, actor candidates, an optional primary actor,
a deterministic decision reason, normalized zone, source type, and optional video
path. JSON fields are encoded in SQLite and decoded by `recent_events`.

Supported event types are `suspected_theft`, `authorized_removal`,
`inventory_recovered`, and `unattributed_inventory_change`. Only a confirmed
`suspected_theft` from a live source activates external alerts. Replay events are
always simulation-only. VLM output never participates in this decision contract.
