"""Self-check: a run must start at the FIRST frame of the video.

Everything downstream (clip times, TOI, the shift timeline, the review overlay)
is timed by counting frames from the first one the pipeline reads. On some files
cv2's "seek to frame 0" returns a frame a whole GOP later while still reporting
position 1 — on the 25 fps clip this was found with, 100 frames (4 s) in. The
pipeline therefore reads from a freshly opened capture and never seeks.

Run: python3 test_videostart.py
"""
import glob
import cv2
import numpy as np


def _sig(frame):
    return cv2.resize(frame, (32, 18)).astype(np.float32)


def _natural_frames(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(_sig(f))
    cap.release()
    return out


def _first_frame_the_pipeline_sees(path):
    """Exactly what process_video_job does before its reader thread starts."""
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    _ = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
         cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f = cap.read()
    cap.release()
    assert ok
    return _sig(f)


def test_processing_starts_on_the_first_frame():
    checked = 0
    for path in sorted(glob.glob("uploads/*.mp4"))[:6]:
        natural = _natural_frames(path, 150)
        if len(natural) < 2:
            continue
        first = _first_frame_the_pipeline_sees(path)
        diffs = [float(np.abs(first - s).mean()) for s in natural]
        at = int(np.argmin(diffs))
        assert at == 0, f"{path}: the run would start at frame {at}, not the first one"
        checked += 1
    if not checked:
        print("test_processing_starts_on_the_first_frame SKIPPED (no videos in uploads/)")
        return
    print(f"test_processing_starts_on_the_first_frame OK ({checked} videos)")





def test_frames_are_timed_by_the_decoder_clock():
    """A frame's time must come from the decoder, not from counting frames.

    A real broadcast cut was found whose first decodable frame sits at 0.96 s:
    counting from zero put every clip, every TOI and the review overlay a second
    ahead of the video a player shows.
    """
    checked = 0
    for path in sorted(glob.glob("uploads/*.mp4"))[:6]:
        cap = cv2.VideoCapture(path)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue
        t0 = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        ts = [t0]
        for _ in range(50):
            ok, _ = cap.read()
            if not ok:
                break
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        cap.release()
        assert all(b > a for a, b in zip(ts, ts[1:])), f"{path}: timestamps not increasing"
        step = (ts[-1] - ts[0]) / (len(ts) - 1)
        assert abs(step - 1 / fps) < 0.005, f"{path}: {step:.4f}s per frame, fps says {1/fps:.4f}"
        checked += 1
    if not checked:
        print("test_frames_are_timed_by_the_decoder_clock SKIPPED (no videos in uploads/)")
        return
    print(f"test_frames_are_timed_by_the_decoder_clock OK ({checked} videos)")

if __name__ == "__main__":
    test_processing_starts_on_the_first_frame()
    test_frames_are_timed_by_the_decoder_clock()
    print("video start OK")
