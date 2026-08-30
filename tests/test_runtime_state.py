from unittest import TestCase

import numpy as np

from src.utils.runtime_state import RuntimeState


class RuntimeStateTests(TestCase):
    def test_metadata_snapshot_omits_expensive_frame_and_perception_payloads(self):
        state = RuntimeState()
        state.update(
            frame=np.ones((10, 10, 3), dtype=np.uint8),
            perception={"items": [{"label": "bottle"}]},
        )

        snapshot = state.snapshot(include_frame=False, include_perception=False)

        self.assertNotIn("frame", snapshot)
        self.assertNotIn("perception", snapshot)
        self.assertIn("security", snapshot)

    def test_single_value_reads_are_isolated_from_shared_state(self):
        state = RuntimeState()
        state.update(device_info={"selected": "cpu", "memory": {"pressure": "normal"}})

        device = state.get("device_info")
        device["memory"]["pressure"] = "high"

        self.assertEqual(state.get("device_info")["memory"]["pressure"], "normal")

    def test_frame_reads_return_an_independent_copy(self):
        state = RuntimeState()
        state.update(frame=np.ones((2, 2, 3), dtype=np.uint8))

        frame = state.get("frame")
        frame[0, 0, 0] = 0

        self.assertEqual(int(state.get("frame")[0, 0, 0]), 1)
