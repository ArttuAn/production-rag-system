"""Required authentication for the paid Gradio UI."""

import os


def launch_options():
    password = os.environ.get("RAG_PASSWORD", "")
    if len(password.strip()) < 32:
        raise RuntimeError("Set a unique RAG_PASSWORD of at least 32 characters before launching")
    username = os.environ.get("RAG_USERNAME", "admin").strip()
    if not username:
        raise RuntimeError("RAG_USERNAME must not be empty")
    return {
        "auth": (username, password),
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        "share": False,
    }
