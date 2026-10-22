"""Desktop entry point for Player Tracking.

Serves the Flask app on a loopback port and shows it in a native window, so the
user never sees a terminal, a URL or a port. Importing `app` drags in torch and
opencv (5-10 s), so the window opens on a loading page first and swaps to the
real UI once the server is up — otherwise a double-click looks like nothing
happened and people click again.
"""
import contextlib
import io
import os
import socket
import subprocess
import sys
import threading
import traceback
from pathlib import Path

TITLE = "Player Tracking"

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
<body><div><div class="mark">PT</div><div class="bar"><i></i></div></div></body></html>"""


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
    """Headless smoke test of a packaged build, offline: the server answers, the
    detector loads and tracks, and the H.264 writer really produces a playable
    file. All three fail silently - opencv off macOS has no avc1 encoder and
    still reports success, a detector that cannot load only shows up mid-job,
    and a model fetched at run time works on a runner but not on a user's
    machine. Run on every platform that gets an installer.
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

    # From here the build has to stand on its own. A CI runner is online, so a
    # model ultralytics quietly fetches mid-job succeeds there and fails on the
    # user's machine - that is how 1.0.1 shipped needing a 6 MB download. Child
    # processes go too: a blocked urllib falls back to curl, and a missing
    # package makes ultralytics run `sys.executable -m pip install`, which in a
    # bundle re-launches the app. Both are refused rather than hidden.
    connect, popen = socket.socket.connect, subprocess.Popen

    def loopback_only(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"selftest is offline on purpose, tried to reach {address}")
        return connect(self, address, *args, **kwargs)

    def no_fetch(args, *rest, **kwargs):
        # Only fetchers and installers are refused: coremltools shells out to
        # sw_vers on every CoreML prediction, and that has to keep working.
        line = " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if any(w in line for w in ("http://", "https://", "curl", "wget", "pip")):
            raise OSError(f"selftest fetches nothing on purpose, tried to run {line}")
        return popen(args, *rest, **kwargs)

    socket.socket.connect, subprocess.Popen = loopback_only, no_fetch

    # Load the detector the way a job does. Shipping the .mlpackage without
    # coremltools passed every earlier check and then failed on the user's
    # first video, so the build is only good if the model it ships actually runs.
    coreml = app_mod.MODELS_DIR / "Player_Detection.mlpackage"
    det = coreml if coreml.exists() else app_mod.MODELS_DIR / "Player_Detection.pt"
    model, used_coreml = app_mod.load_detector(str(det), "cpu")
    assert used_coreml == coreml.exists(), "bundled .mlpackage did not load"
    if used_coreml:
        # BoT-SORT cannot read features out of a CoreML detector and needs this
        # encoder; without it ultralytics downloads one mid-job.
        assert app_mod.MODELS_DIR.joinpath("yolo26n-cls.pt").exists(), "ReID encoder missing"
    # One tracked frame, not just a detection: the tracker builds its ReID
    # encoder and its assignment solver on the first call, and both have failed
    # in a packaged build while plain detection kept working.
    tracker = app_mod.effective_tracker(str(app_mod.TRACKERS_DIR / "tracker_stable.yaml"),
                                        Path(tempfile.mkdtemp()), 25.0, used_coreml)
    model.track(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640,
                tracker=tracker, persist=True, verbose=False)
    print(f"  detector tracks (CoreML: {used_coreml})")

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
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = selftest()
        except Exception:
            out.write(traceback.format_exc())
            code = 1
        # A windowed build has no console: on Windows nothing we print reaches
        # the CI log, so the whole run goes to a file the workflow reads out.
        try:
            (Path(sys.executable).parent / "psf_selftest.log").write_text(
                out.getvalue(), encoding="utf-8")
        except OSError:
            pass
        print(out.getvalue())
        return code
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
