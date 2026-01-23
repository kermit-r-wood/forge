# -*- mode: python ; coding: utf-8 -*-

import sys
from PySide6 import QtCore
import os

block_cipher = None

# Defines the root of the project
project_root = os.path.abspath(os.path.join(os.getcwd(), 'src'))

a = Analysis(
    [os.path.join(project_root, 'forge', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=['sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Forge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to False for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Forge',
)
