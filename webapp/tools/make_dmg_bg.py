"""Render the DMG window background: the install step and the one-time
Gatekeeper walkthrough, shown where the user actually is when it bites them."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "packaging"
W, H = 760, 560
BG, CARD, FG, MUTED, ORANGE = "#101417", "#191F24", "#FFFFFF", "#9AA4AD", "#FF6300"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((W / 2, 46), "Player Scouting & Fitness", font=font(BOLD, 26), fill=FG, anchor="mm")
    d.text((W / 2, 78), "Drag the icon into the Applications folder",
           font=font(REG, 15), fill=MUTED, anchor="mm")

    # arrow between the two icon slots (icons themselves are placed by create-dmg)
    d.line([(300, 200), (455, 200)], fill=ORANGE, width=3)
    d.polygon([(455, 190), (480, 200), (455, 210)], fill=ORANGE)

    d.rounded_rectangle([40, 300, W - 40, H - 36], radius=8, fill=CARD, outline="#313B45")
    d.text((70, 330), "First launch \u2014 once only", font=font(BOLD, 17), fill=ORANGE)
    steps = [
        "1.  Open Applications and double-click PS&F",
        "2.  macOS says it cannot verify the developer \u2014 click Done",
        "3.  Open System Settings \u2192 Privacy & Security",
        "4.  Scroll down, click \u201cOpen Anyway\u201d, then \u201cOpen\u201d",
    ]
    y = 366
    for s in steps:
        d.text((70, y), s, font=font(REG, 14), fill=FG)
        y += 28
    d.text((70, y + 6), "After that the app opens with a normal double-click.",
           font=font(REG, 14), fill=MUTED)

    OUT.mkdir(exist_ok=True)
    img.save(OUT / "dmg_bg.png")
    img.resize((W * 2, H * 2), Image.LANCZOS).save(OUT / "dmg_bg@2x.png")
    print("dmg background ->", OUT / "dmg_bg.png")


if __name__ == "__main__":
    main()
