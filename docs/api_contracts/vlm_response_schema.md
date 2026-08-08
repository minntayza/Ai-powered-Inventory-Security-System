# VLM response contract

The VQA pipeline returns `ok`, `answer`, `error`, `source_id`, and, on success,
`cached` and an ISO timestamp. `source_id` is `live` or the incident event UUID.
VLM output is descriptive and must never change authorization or security state.

