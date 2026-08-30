from unittest import TestCase, mock

from src.utils.gpu_manager import (
    device_info,
    gpu_memory_status,
    release_accelerator_cache,
    resolve_torch_device,
)


class DeviceSelectionTests(TestCase):
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_auto_selects_cuda_when_available(self, _available):
        self.assertEqual(resolve_torch_device("auto"), "cuda:0")

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_selects_cpu_without_accelerator(self, _mps, _cuda):
        self.assertEqual(resolve_torch_device("auto"), "cpu")

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_selects_mps_on_apple_silicon(self, _mps, _cuda):
        self.assertEqual(resolve_torch_device("auto"), "mps")

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_falls_back_to_cpu(self, _available):
        self.assertEqual(resolve_torch_device("cuda:0"), "cpu")

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    def test_explicit_cuda_falls_back_to_mps_when_available(self, _mps, _cuda):
        self.assertEqual(resolve_torch_device("cuda:0"), "mps")

    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_explicit_mps_falls_back_to_cpu(self, _available):
        self.assertEqual(resolve_torch_device("mps"), "cpu")

    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_explicit_cpu_is_preserved(self, _available):
        self.assertEqual(resolve_torch_device("cpu"), "cpu")


class GpuMemoryStatusTests(TestCase):
    def test_cuda_memory_pressure_is_reported(self):
        gib = 1024 ** 3
        torch = mock.Mock()
        torch.cuda.get_device_properties.return_value.total_memory = 8 * gib
        torch.cuda.memory_allocated.return_value = 2 * gib
        torch.cuda.memory_reserved.return_value = 3 * gib

        result = gpu_memory_status("cuda:0", torch_module=torch)

        self.assertEqual(
            result,
            {
                "available": True,
                "device": "cuda:0",
                "allocated_gb": 2.0,
                "reserved_gb": 3.0,
                "total_gb": 8.0,
                "utilization": 0.375,
                "pressure": "normal",
            },
        )

    def test_cuda_cache_can_be_released(self):
        torch = mock.Mock()

        released = release_accelerator_cache("cuda:0", torch_module=torch)

        self.assertEqual(
            {"released": released, "calls": torch.cuda.empty_cache.call_count},
            {"released": True, "calls": 1},
        )

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.memory_allocated", return_value=6 * 1024 ** 3)
    @mock.patch("torch.cuda.memory_reserved", return_value=7 * 1024 ** 3)
    @mock.patch("torch.cuda.get_device_properties")
    def test_device_info_surfaces_high_memory_pressure(
        self, properties, _reserved, _allocated, _available
    ):
        properties.return_value.name = "Demo GPU"
        properties.return_value.total_memory = 8 * 1024 ** 3

        result = device_info("cuda:0")

        self.assertEqual(result["memory"]["pressure"], "high")
