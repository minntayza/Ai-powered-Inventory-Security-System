import sys
from pathlib import Path
from unittest import TestCase


class DashboardEntrypointTests(TestCase):
    def test_entrypoint_adds_project_root_before_local_imports(self):
        project_root = Path(__file__).resolve().parents[1]
        app_path = project_root / "src" / "module_c_ui_dashboard" / "app.py"
        original_path = list(sys.path)
        try:
            sys.path[:] = [str(app_path.parent)] + [
                value for value in sys.path if value != str(project_root)
            ]
            source = app_path.read_text(encoding="utf-8")
            bootstrap = source.split("import streamlit as st", 1)[0]
            namespace = {"__file__": str(app_path), "__name__": "dashboard_bootstrap_test"}
            exec(compile(bootstrap, str(app_path), "exec"), namespace)
            self.assertEqual(Path(sys.path[0]), project_root)
        finally:
            sys.path[:] = original_path
