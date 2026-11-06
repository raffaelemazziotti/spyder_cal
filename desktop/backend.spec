from pathlib import Path
import importlib.metadata
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files('psychopy', excludes=['tests/**', 'demos/**'])
datas += collect_data_files('libusb_package')
for package in ['psychopy', 'pyglet', 'freetype-py', 'python-bidi', 'arabic-reshaper']:
    datas += copy_metadata(package)
for distribution in importlib.metadata.distributions():
    for file in distribution.files or []:
        if '.dist-info/' in str(file).replace('\\', '/') and any(word in file.name.lower() for word in ('license', 'copying', 'notice')):
            datas.append((str(distribution.locate_file(file)), str(Path('licenses') / distribution.metadata['Name'] / file.parent)))
binaries = collect_dynamic_libs('libusb_package')
a = Analysis(
    [str(root / 'desktop_backend.py')], pathex=[str(root)],
    binaries=binaries, datas=datas,
    hiddenimports=['psychopy.visual.backends.pygletbackend', 'psychopy.visual.rect',
                   'psychopy.visual.polygon', 'psychopy.visual.shape', 'psychopy.visual.basevisual',
                   'psychopy.visual.grating', 'psychopy.visual.slider', 'psychopy.visual.line',
                   'pyglet.gl', 'pyglet.window', 'pyglet.canvas', 'pyglet.font', 'pyglet.text',
                   'usb.backend.libusb1', 'matplotlib.backends.backend_agg', 'matplotlib.backends.backend_svg'],
    excludes=['PyQt6', 'PyQt5', 'PySide6', 'PySide2', 'wx', 'tkinter',
              'psychopy.app', 'psychopy.iohub', 'psychopy.sound', 'psychopy.microphone',
              'psychopy.tests', 'psychopy.demos', 'cv2', 'ffpyplayer', 'moviepy',
              'pyarrow', 'tables', 'IPython', 'pytest'],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[str(root / 'desktop' / 'runtime-hook.py')],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='spyder-cal-backend',
          console=True, debug=False, strip=False, upx=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='spyder-cal-backend')
