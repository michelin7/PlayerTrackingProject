"""Self-check for team brightness clustering (PlayerIdentifier._update_team_centers).

The split point must stay near the true middle of the two jersey-brightness
clusters even when one team is on screen far more than the other — a single
running mean would drift toward the majority and misclassify the minority.
Run: python3 test_team.py
"""
import app
import numpy as np


class _Boxes:
    """Minimal stand-in for a YOLO Boxes object (len / .xyxy / .conf)."""
    def __init__(self, xyxy, conf):
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
    def __len__(self):
        return len(self.conf)


def test_ice_excluded_dark_jersey():
    pi = app.PlayerIdentifier.__new__(app.PlayerIdentifier)
    # dark jersey (BGR) with a bright, low-sat ICE border ring
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[:, :] = (120, 45, 20)                 # dark blue-ish jersey, V=120 (not ice)
    crop[:10, :] = crop[-10:, :] = (245, 245, 245)   # ice ring (bright, low-sat)
    crop[:, :10] = crop[:, -10:] = (245, 245, 245)
    b = pi._jersey_brightness(crop)
    assert b < 150, ("ice border must be excluded -> dark", b)  # naive median would be bright

    # a genuine WHITE jersey (all bright, low-sat) must still read bright
    white = np.full((40, 40, 3), 245, dtype=np.uint8)
    assert pi._jersey_brightness(white) > 200, "white jersey stays light"
    print("test_ice_excluded_dark_jersey OK")


def test_number_polarity_team():
    pi = app.PlayerIdentifier.__new__(app.PlayerIdentifier)
    pi._light_team, pi._dark_team = "Home", "Away"
    # LIGHT jersey (bright) with a DARK digit -> light team
    crop = np.full((40, 40, 3), 235, dtype=np.uint8)   # RGB bright jersey
    crop[15:25, 15:25] = 25                             # dark digit
    team, strength = pi._number_team(crop, _Boxes([[15, 15, 25, 25]], [0.9]))
    assert team == "Home" and strength > 0, (team, strength)

    # DARK jersey with a LIGHT digit -> dark team
    crop2 = np.full((40, 40, 3), 30, dtype=np.uint8)
    crop2[15:25, 15:25] = 240
    team2, _ = pi._number_team(crop2, _Boxes([[15, 15, 25, 25]], [0.9]))
    assert team2 == "Away", team2

    # no boxes -> no vote
    assert pi._number_team(crop, _Boxes([], [])) == (None, 0.0)
    print("test_number_polarity_team OK")


def _make(light=215.0, dark=180.0):
    pi = app.PlayerIdentifier.__new__(app.PlayerIdentifier)
    pi._light_center = light
    pi._dark_center = dark
    return pi


def test_midpoint_stable_under_imbalance():
    pi = _make()
    mid = 0.0
    # 90% dark crops (V~175), 10% light (V~215)
    for b in [175.0] * 180 + [215.0] * 20:
        mid = pi._update_team_centers(b)
    # midpoint tracks the middle of the two clusters, not the crowd mean (~179)
    assert 188 <= mid <= 202, mid
    # a light player at 200 is above the split; a dark player at 190 below it
    assert 200 >= mid and 190 < mid
    # clusters never collapse
    assert pi._light_center - pi._dark_center >= app.PlayerIdentifier.TEAM_MIN_GAP


def test_centers_do_not_collapse_on_midtone_noise():
    pi = _make()
    for b in [197.0] * 100:  # a stream of ambiguous mid-tone crops
        pi._update_team_centers(b)
    assert pi._light_center - pi._dark_center >= app.PlayerIdentifier.TEAM_MIN_GAP


if __name__ == "__main__":
    test_ice_excluded_dark_jersey()
    test_number_polarity_team()
    test_midpoint_stable_under_imbalance()
    test_centers_do_not_collapse_on_midtone_noise()
    print("team clustering OK")
