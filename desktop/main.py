"""
PyWebView entrypoint. Boots the FastAPI server on a random local port,
then opens a chromeless desktop window pointing at it.
"""
from __future__ import annotations
import os
import sys
import socket
import threading
import time

import uvicorn
import webview

from .api import app, set_pywebview_window
from .settings import SETTINGS_PATH, load_settings


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.run()


class JsBridge:
    """Methods callable from JS via window.pywebview.api.*"""
    def pick_folder(self) -> str | None:
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def open_settings_file(self) -> str:
        return SETTINGS_PATH

    def quit(self):
        webview.windows[0].destroy()


def main():
    # ensure settings file exists
    load_settings()

    port = _free_port()
    t = threading.Thread(target=_run_server, args=(port,), daemon=True)
    t.start()

    # wait for server to come up (cheap retry)
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)

    url = f"http://127.0.0.1:{port}/"
    bridge = JsBridge()
    win = webview.create_window(
        "AI File Explorer",
        url,
        js_api=bridge,
        width=1180,
        height=780,
        min_size=(880, 560),
        background_color="#14140f",   # graphite-0 from the design system
        text_select=True,
    )
    set_pywebview_window(win)

    # gui="cocoa" on mac, "edgechromium" on windows, "qt"/"gtk" on linux — auto by default
    webview.start(debug=False)


if __name__ == "__main__":
    main()
