import os
import unittest
from unittest.mock import patch

from src.access import launch_options


class AccessTests(unittest.TestCase):
    def test_missing_short_and_whitespace_passwords_prevent_launch(self):
        for password in ["", "short", " " * 64]:
            with patch.dict(os.environ, {"RAG_PASSWORD": password}):
                with self.assertRaises(RuntimeError):
                    launch_options()

    def test_authentication_is_required_and_sharing_disabled(self):
        with patch.dict(os.environ, {"RAG_PASSWORD": "p" * 32, "RAG_USERNAME": "owner"}, clear=True):
            options = launch_options()
        self.assertEqual(options["auth"], ("owner", "p" * 32))
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertFalse(options["share"])

    def test_remote_binding_keeps_authentication(self):
        with patch.dict(os.environ, {"RAG_PASSWORD": "p" * 32, "GRADIO_SERVER_NAME": "0.0.0.0"}, clear=True):
            self.assertEqual(launch_options()["auth"][1], "p" * 32)
