"""Refresh the web assets the app serves itself: Chart.js and the Google Fonts.

Both are render-blocking, so fetching them from a CDN at run time leaves the
desktop window blank until the request times out on a machine with no network.
Run from webapp/ after changing a version or a font weight:

    python tools/fetch_assets.py
"""
import hashlib
import pathlib
import re
import urllib.request

CHART = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700"
         "&family=Barlow+Condensed:wght@500;600;700"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")
# Google serves woff2 only to a browser it recognises; anything else gets ttf.
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"}

HERE = pathlib.Path(__file__).resolve().parent.parent      # webapp/


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def main():
    (HERE / "static" / "vendor").mkdir(parents=True, exist_ok=True)
    fonts_dir = HERE / "static" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    chart = get(CHART)
    (HERE / "static" / "vendor" / "chart.umd.min.js").write_bytes(chart)
    print(f"chart.umd.min.js {len(chart) // 1024} KB")

    css = get(FONTS).decode("utf-8")
    urls = sorted(set(re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)))
    total = 0
    for url in urls:
        # The upstream names collide across families, so key each file by its url.
        name = hashlib.md5(url.encode()).hexdigest()[:8] + "-" + url.rsplit("/", 1)[-1]
        data = get(url)
        (fonts_dir / name).write_bytes(data)
        css = css.replace(url, "../fonts/" + name)
        total += len(data)
    (HERE / "static" / "css" / "fonts.css").write_text(
        "/* Google Fonts, served from the app: a blocking request to the network "
        "leaves\n   the window blank until it times out, and an installed app has to "
        "render\n   offline. Regenerate with tools/fetch_assets.py. */\n" + css,
        encoding="utf-8")
    print(f"{len(urls)} font files, {total // 1024} KB")


if __name__ == "__main__":
    main()
