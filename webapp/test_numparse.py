"""Self-check for jersey-number parsing (PlayerIdentifier._parse_number_results).

Covers the 1-digit vs 2-digit decision, faint-second-digit recovery (#44 with a
weak trailing 4), roster inference for a lone digit, and noise rejection.
Run: python3 test_numparse.py
"""
import types

import app


class _Boxes:
    def __init__(self, digs):
        self.xyxy = types.SimpleNamespace(tolist=lambda: [[d[0], 0, d[1], 40] for d in digs])
        self.cls = types.SimpleNamespace(tolist=lambda: [float(d[2]) for d in digs])
        self.conf = types.SimpleNamespace(tolist=lambda: [d[3] for d in digs])

    def __len__(self):
        return len(self.xyxy.tolist())


class _Res:
    def __init__(self, digs):
        self.boxes = _Boxes(digs)
        self.names = {i: str(i) for i in range(10)}


def parse(digs, squad, cw=100, sink=None):
    """digs: list of (x1, x2, digit, conf)."""
    pi = app.PlayerIdentifier.__new__(app.PlayerIdentifier)
    pi.Home = squad
    pi.Away = squad
    pi._num_debug = sink if sink is not None else []
    pi._num_debug_on = sink is not None
    pi._num_debug_cap = 1000
    pi._last_guess = {}
    return pi._parse_number_results([_Res(digs)], "Home", crop_width=cw, pid=7, frame=1)


def test_number_parsing():
    # #44 with a faint trailing "4" -> recovered as 44 (was: 4)
    assert parse([(30, 50, 4, 0.60), (55, 70, 4, 0.22)], {"44": "A"})[1] == 44
    # genuine centered single digit stays single
    assert parse([(38, 58, 4, 0.70)], {"4": "B"})[1] == 4
    # lone digit with no second detection is NOT guessed into a 2-digit number
    assert parse([(10, 28, 4, 0.55)], {"44": "C"})[1] is None
    # faint lone noise -> Unknown
    assert parse([(40, 55, 7, 0.20)], {"44": "D", "7": "E"})[1] is None
    # two strong digits -> two-digit number
    assert parse([(25, 45, 9, 0.7), (50, 70, 0, 0.6)], {"90": "F"})[1] == 90
    # ambiguous roster (4/14/24) with a lone strong 4 -> keep single 4
    assert parse([(40, 60, 4, 0.7)], {"4": "G", "14": "H", "24": "I"})[1] == 4

    # debug log records the raw detections + reason for a decision
    sink = []
    parse([(30, 50, 4, 0.60), (55, 70, 4, 0.22)], {"44": "A"}, sink=sink)
    assert len(sink) == 1 and sink[0]["num"] == 44
    assert [4, 0.6] == sink[0]["dets"][0][:2]  # YOLO's raw digit + conf logged
    print("number parsing OK")


if __name__ == "__main__":
    test_number_parsing()
