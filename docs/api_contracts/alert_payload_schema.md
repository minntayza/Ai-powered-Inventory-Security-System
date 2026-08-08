# Alert payload contract

Telegram receives a confirmed security event plus its saved incident JPEG. Delivery
returns `status` (`sent`, `failed`, or `disabled`) and a human-readable `detail`.
Failures are written to `alert_attempts` and never stop local monitoring.

