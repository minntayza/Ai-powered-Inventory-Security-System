from unittest import TestCase

from src.utils.performance_monitor import PerformanceMonitor


class PerformanceMonitorTests(TestCase):
    def test_samples_and_frame_counters_are_bounded(self):
        monitor = PerformanceMonitor(sample_count=3)
        for value in range(10):
            monitor.record("yolo_ms", value)
        monitor.capture_tick(1)
        monitor.capture_tick(2)
        monitor.skip(2)
        result = monitor.snapshot()
        self.assertEqual(result["yolo_ms_avg"], 8.0)
        self.assertEqual(result["capture_fps"], 1.0)
        self.assertEqual(result["skipped_frames"], 2)
