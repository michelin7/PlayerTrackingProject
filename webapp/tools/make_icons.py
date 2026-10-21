"""Render the app icon (.icns for macOS, .ico for Windows) from the Player Tracking mark."""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "packaging"
BG, FG = "#101417", "#FF6300"
FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"


def render(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(size * 0.22)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG,
                        outline="#313B45", width=max(1, size // 128))
    try:
        font = ImageFont.truetype(FONT, int(size * 0.42))
    except OSError:
        font = ImageFont.load_default()
    d.text((size / 2, size * 0.46), "PT", font=font, fill=FG, anchor="mm")
    d.rectangle([size * 0.28, size * 0.68, size * 0.72, size * 0.71], fill=FG)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    render(1024).save(OUT / "icon.png")
    render(256).save(OUT / "icon.ico",
                     sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    if sys.platform == "darwin":
        iconset = OUT / "icon.iconset"
        iconset.mkdir(exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            render(s).save(iconset / f"icon_{s}x{s}.png")
            render(s * 2).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(OUT / "icon.icns")], check=True)
    print("icons ->", OUT)


if __name__ == "__main__":
    main()
