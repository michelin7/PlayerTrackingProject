# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PS&F. Build from webapp/: pyinstaller packaging/psf.spec"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

APP = Path(SPECPATH).resolve().parent          # webapp/
ROOT = APP.parent                              # project root

datas = [
    (str(APP / "templates"), "templates"),
    (str(APP / "static"), "static"),
    (str(ROOT / "trackers"), "trackers"),
    (str(ROOT / "models" / "Player_Detection.pt"), "models"),
    (str(ROOT / "models" / "NumbersRec.pt"), "models"),
]
if sys.platform == "darwin":                   # CoreML detector, macOS only
    datas.append((str(ROOT / "models" / "Player_Detection.mlpackage"),
                  "models/Player_Detection.mlpackage"))

binaries, hiddenimports = [], collect_submodules("sklearn.utils")

# ultralytics by data files only: collecting every submodule drags in its
# optional exporters (onnx, openvino, benchmarks) and their dependencies.
# The parts loaded dynamically at track time are named explicitly instead.
datas += collect_data_files("ultralytics")
for mod in ("ultralytics.trackers", "ultralytics.nn", "ultralytics.models",
            "ultralytics.utils"):
    hiddenimports += collect_submodules(mod)

for pkg in ("torch", "torchvision"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(APP / "launcher.py")],
    pathex=[str(APP)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython",
              "jupyter", "notebook", "tensorflow", "onnxruntime", "pytest",
              # dragged in from the dev environment, unused by the app
              "polars", "playwright", "numba", "llvmlite", "skimage",
              "imageio", "imageio_ffmpeg", "statsmodels", "patsy", "sympy",
              "panel", "holoviews", "datashader",
              "onnx", "openvino", "tensorboard", "dask", "bokeh", "zmq"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PSF",
          console=False, icon=str(APP / "packaging" /
                                  ("icon.icns" if sys.platform == "darwin" else "icon.ico")))

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PSF")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="Player Scouting & Fitness.app",
                 icon=str(APP / "packaging" / "icon.icns"),
                 bundle_identifier="ru.susu.psf",
                 info_plist={
                     "CFBundleShortVersionString": "1.0.0",
                     "NSHighResolutionCapable": True,
                     "LSMinimumSystemVersion": "11.0",
                     "NSCameraUsageDescription": "",
                 })
