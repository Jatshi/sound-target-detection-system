# PyInstaller spec template. Build from the app root:
# pyinstaller packaging/sounddet.spec

block_cipher = None

a = Analysis(
    ["scripts/run_desktop.py"],
    pathex=["."],
    binaries=[],
    datas=[("configs", "configs")],
    hiddenimports=["sounddet"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SoundTargetDetection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
