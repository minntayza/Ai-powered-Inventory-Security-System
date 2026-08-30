import base64
from importlib import import_module
from unittest import TestCase
from unittest.mock import patch

import numpy as np


class LiveFrameRenderingTests(TestCase):
    def test_inventory_status_table_is_rendered_for_the_operator(self):
        operational_view = import_module(
            "src.module_c_ui_dashboard.operational_view"
        )

        with patch.object(operational_view.st, "dataframe") as dataframe:
            operational_view.render_inventory_status(
                {"bottle": 1},
                {"bottle": 2},
            )

        dataframe.assert_called_once_with(
            [
                {
                    "Item": "Bottle",
                    "Current": 1,
                    "Baseline": 2,
                    "Difference": -1,
                    "Status": "Missing (1)",
                }
            ],
            hide_index=True,
            use_container_width=True,
        )

    def test_inventory_status_rows_explain_present_missing_and_extra_items(self):
        operational_view = import_module(
            "src.module_c_ui_dashboard.operational_view"
        )

        rows = operational_view.inventory_status_rows(
            {"bottle": 1, "laptop": 1, "scissors": 2},
            {"bottle": 2, "laptop": 1, "scissors": 1},
        )

        self.assertEqual(
            rows,
            [
                {
                    "Item": "Bottle",
                    "Current": 1,
                    "Baseline": 2,
                    "Difference": -1,
                    "Status": "Missing (1)",
                },
                {
                    "Item": "Laptop",
                    "Current": 1,
                    "Baseline": 1,
                    "Difference": 0,
                    "Status": "Present",
                },
                {
                    "Item": "Scissors",
                    "Current": 2,
                    "Baseline": 1,
                    "Difference": 1,
                    "Status": "Discrepancy (+1)",
                },
            ],
        )

    def test_live_frame_uses_embedded_jpeg_instead_of_temporary_media_url(self):
        operational_view = import_module(
            "src.module_c_ui_dashboard.operational_view"
        )
        encode_frame_data_uri = getattr(
            operational_view, "encode_frame_data_uri", lambda frame: None
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        uri = encode_frame_data_uri(frame)

        self.assertIsInstance(uri, str)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        jpeg = base64.b64decode(uri.split(",", 1)[1])
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        self.assertEqual(jpeg[-2:], b"\xff\xd9")

    def test_cuda_status_warns_when_memory_pressure_is_high(self):
        operational_view = import_module(
            "src.module_c_ui_dashboard.operational_view"
        )

        status = operational_view.format_device_status(
            {
                "selected": "cuda:0",
                "gpu_name": "Demo GPU",
                "gpu_memory_gb": 8.0,
                "memory": {
                    "reserved_gb": 7.0,
                    "total_gb": 8.0,
                    "pressure": "high",
                },
            }
        )

        self.assertEqual(
            status,
            "Inference: cuda:0 — Demo GPU (7.0/8.0 GB reserved) — HIGH MEMORY PRESSURE",
        )

    def test_degraded_face_health_has_operator_message(self):
        operational_view = import_module(
            "src.module_c_ui_dashboard.operational_view"
        )

        messages = operational_view.component_health_messages(
            {
                "face": {
                    "healthy": False,
                    "last_error": "face backend failed",
                }
            }
        )

        self.assertEqual(
            messages,
            ["Face recognition degraded: face backend failed"],
        )
