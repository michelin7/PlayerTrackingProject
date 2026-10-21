"""Desktop entry point for PS&F.

Serves the Flask app on a loopback port and shows it in a native window, so the
user never sees a terminal, a URL or a port. Importing `app` drags in torch and
opencv (5-10 s), so the window opens on a loading page first and swaps to the
real UI once the server is up — otherwise a double-click looks like nothing
happened and people click again.
"""
import os
import socket
import subprocess
import sys
import threading
import traceback

TITLE = "Player Scouting & Fitness"

LOADING_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html,body{margin:0;height:100%;background:#101417;display:flex;
            align-items:center;justify-content:center}
  .mark{font:700 64px/1 Impact,'Barlow Condensed',sans-serif;letter-spacing:2px;
        color:#FF6300;text-align:center}
  .bar{width:180px;height:3px;margin:28px auto 0;background:#232A31;overflow:hidden}
  .bar i{display:block;width:60px;height:3px;background:#FF6300;
         animation:slide 1.1s infinite ease-in-out}
  @keyframes slide{0%{transform:translateX(-60px)}100%{transform:translateX(180px)}}
</style></head>
<body><div><div class="mark">PS&amp;F</div><div class="bar"><i></i></div></div></body></html>"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def native_error(text):
    """Report a startup failure when there is no window left to show it in."""
    if sys.platform == "darwin":
        msg = text.replace('"', "'")
        subprocess.run(["osascript", "-e",
                        f'display dialog "{msg}" with title "{TITLE}" '
                        'buttons {"OK"} with icon stop'], check=False)
    elif sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10)
    else:
        print(text, file=sys.stderr)


def busy(app_mod):
    """True while a job or a live session would lose work on close."""
    if any(j.get("status") in ("processing", "finalizing")
           for j in app_mod.processing_jobs.values()):
        return True
    return getattr(app_mod, "live_session", None) is not None


def boot(window):
    try:
        import app as app_mod
        from werkzeug.serving import make_server

        port = free_port()
        server = make_server("127.0.0.1", port, app_mod.app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def closing():
            if not busy(app_mod):
                return True
            return window.create_confirmation_dialog(
                TITLE, "Video processing is still running. Close the window and lose the result?")

        window.events.closing += closing
        window.load_url(f"http://127.0.0.1:{port}/")
    except Exception:
        native_error("Could not start the application.\n\n" + traceback.format_exc(limit=3))
        os._exit(1)


def selftest():
    """Headless smoke test of a packaged build: the server answers and the
    H.264 writer really produces a playable file. The writer is the one that
    fails silently off macOS - opencv there has no avc1 encoder and still
    reports success, leaving every player clip empty.
    """
    import tempfile
    import urllib.request

    import cv2
    import numpy as np
    from werkzeug.serving import make_server

    import app as app_mod

    port = free_port()
    server = make_server("127.0.0.1", port, app_mod.app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for path in ("/", "/analysis", "/config", "/results"):
        code = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30).getcode()
        assert code == 200, f"{path} -> {code}"
        print(f"  {path} {code}")

    assert app_mod.MODELS_DIR.joinpath("Player_Detection.pt").exists(), "detector missing"
    assert app_mod.MODELS_DIR.joinpath("NumbersRec.pt").exists(), "number model missing"
    assert app_mod.TRACKERS_DIR.joinpath("tracker.yaml").exists(), "tracker config missing"

    clip = Path(tempfile.gettempdir()) / "psf_selftest.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"avc1"), 25, (320, 240))
    assert writer.isOpened(), "VideoWriter did not open"
    rng = np.random.default_rng(0)
    for _ in range(25):
        # noise, not flat colour: a broken encoder and a well-compressed clip
        # are the same handful of bytes otherwise
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    assert clip.exists() and clip.stat().st_size > 5000, f"empty clip: {clip}"
    ok, _ = cv2.VideoCapture(str(clip)).read()
    assert ok, "written clip cannot be read back"
    print(f"  H.264 clip {clip.stat().st_size} bytes, readable")
    clip.unlink(missing_ok=True)
    print("selftest OK")
    return 0


def main():
    if "--selftest" in sys.argv:
        from pathlib import Path
        globals()["Path"] = Path
        try:
            return selftest()
        except Exception:
            # A windowed build has no console, so CI would see only an exit code.
            log = Path(sys.executable).parent / "psf_selftest.log"
            log.write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), file=sys.stderr)
            return 1
    try:
        import webview
    except ImportError:
        native_error("pywebview is not installed: pip install pywebview")
        return 1
    window = webview.create_window(TITLE, html=LOADING_HTML,
                                   width=1440, height=900, min_size=(1100, 700))
    webview.start(boot, window)
    # Job threads are daemons and cv2 writers can hang on teardown; leaving the
    # process alive would keep a dead icon in the dock.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
