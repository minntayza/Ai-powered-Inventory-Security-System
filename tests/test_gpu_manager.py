from unittest import TestCase, mock

from src.utils.gpu_manager import resolve_torch_device


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

    @mock.patch("torch.backends.mps.is_available", return_value=False)
    def test_explicit_mps_falls_back_to_cpu(self, _available):
        self.assertEqual(resolve_torch_device("mps"), "cpu")

    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_explicit_cpu_is_preserved(self, _available):
        self.assertEqual(resolve_torch_device("cpu"), "cpu")
