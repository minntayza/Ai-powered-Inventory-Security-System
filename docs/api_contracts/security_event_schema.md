# Security event contract

Events have a UUID `event_id`, ISO timestamp, `event_type`, status, before/after
counts, removed item quantities, associated track IDs, observed authorization
states, optional snapshot path, Telegram delivery status, and acknowledgement flag.

Supported event types are `suspected_theft`, `authorized_removal`,
`inventory_recovered`, and `unattributed_inventory_change`. Only a confirmed
`suspected_theft` activates external alerts.

