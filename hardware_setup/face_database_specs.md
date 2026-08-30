# Face Enrollment Specifications

Face images are biometric data. Collect them with consent, restrict access to the project laptop, review OneDrive/cloud synchronization, and delete them when the demonstration no longer needs them.

## Accepted enrollment

- 1–5 JPG or PNG images per person; 3 varied images are recommended.
- Exactly one clearly visible face in every image.
- A safe display name containing letters, numbers, spaces, `_`, or `-`.
- Face width of at least 50 pixels, matching `person_tracking.min_face_size` in `configs/thresholds.yaml`.
- No masks, heavy blur, severe glare, or face-obscuring accessories for the main enrollment set.

Use the dashboard enrollment form, which validates file type, image count, name safety, and face count. Stored images are organized under:

```text
assets/known_faces/<person-name>/
```

These images are ignored by Git, but local backup or synchronization software may still copy them.

## Recommended photo set

1. Front-facing, neutral expression.
2. Slight turn to the left.
3. Slight turn to the right.
4. Optional: typical exhibition lighting.
5. Optional: typical standing distance, while keeping the face clear.

Avoid using near-identical frames from the same burst as the entire set. Variation helps the demo cover normal pose and lighting changes.

## Camera conditions

- Put soft light in front of the person rather than behind them.
- Position the camera close to eye level.
- Keep the face sharp and large enough at the real approach distance.
- Re-enroll after a major appearance change if recognition becomes unreliable.

## Validation and deletion

After enrollment, test the authorized person from the live camera and confirm an unknown person remains unknown. Recognition uses the configured VGG-Face model, OpenCV detector, cosine distance, and threshold `0.4`; tune only with documented test results.

Use the dashboard's explicit deletion confirmation to remove a person. Verify the directory is gone and that a new live frame no longer recognizes the deleted identity.
