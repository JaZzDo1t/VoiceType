# -*- mode: python ; coding: utf-8 -*-
"""
VoiceType - PyInstaller Specification (ONNX version)
Конфигурация для сборки приложения БЕЗ PyTorch.

Использует RUPunct ONNX для пунктуации вместо Silero TE.
Размер билда: ~500 MB (вместо 4+ GB)
"""

import sys
import site
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Корневая директория проекта
ROOT_DIR = Path(SPECPATH).parent

block_cipher = None

# Ищем путь к site-packages для vosk DLLs
def find_vosk_dlls():
    """Найти DLL файлы vosk в site-packages."""
    vosk_dlls = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        vosk_dir = Path(sp) / 'vosk'
        if vosk_dir.exists():
            for dll in vosk_dir.glob('*.dll'):
                vosk_dlls.append((str(dll), 'vosk'))
            break
    return vosk_dlls

# Бинарные файлы для включения (DLLs)
binaries = find_vosk_dlls()

# Собираем данные для включения
datas = [
    # Иконки и ресурсы
    (str(ROOT_DIR / 'resources' / 'icons'), 'resources/icons'),
    (str(ROOT_DIR / 'resources' / 'sounds'), 'resources/sounds'),

    # Модели Vosk (распознавание речи)
    (str(ROOT_DIR / 'models' / 'vosk-model-small-ru-0.22'), 'models/vosk-model-small-ru-0.22'),
    (str(ROOT_DIR / 'models' / 'vosk-model-ru-0.42'), 'models/vosk-model-ru-0.42'),

    # RUPunct ONNX модель (пунктуация) - НЕ silero-te!
    # Модель скачивается автоматически в ~/.cache/voicetype/rupunct_onnx/
]

# Фильтруем несуществующие пути
datas = [(src, dst) for src, dst in datas if Path(src).exists()]

# Данные для transformers (токенизатор)
datas += collect_data_files('transformers', include_py_files=False)

# Скрытые импорты
hiddenimports = [
    # Vosk
    'vosk',

    # ONNX Runtime (для RUPunct)
    'onnxruntime',

    # Transformers (токенизатор для RUPunct)
    'transformers',
    'transformers.models',
    'transformers.models.bert',
    'transformers.models.bert.tokenization_bert',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'tokenizers',

    # Huggingface Hub
    'huggingface_hub',

    # NumPy
    'numpy',

    # PyAudio
    'pyaudio',

    # pynput - платформо-специфичные модули (Windows)
    'pynput',
    'pynput.keyboard',
    'pynput.keyboard._win32',
    'pynput.mouse',
    'pynput.mouse._win32',

    # PyQt6
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',

    # Loguru
    'loguru',

    # PyYAML
    'yaml',

    # pyperclip (для clipboard)
    'pyperclip',

    # psutil (для мониторинга системы)
    'psutil',

    # Стандартные библиотеки
    'queue',
    'threading',
    'json',
    'sqlite3',
    'wave',
    'struct',
    'pathlib',
    'typing_extensions',
    'packaging',
    'packaging.version',
    'regex',
    'filelock',
    'safetensors',

    # Наши модули
    'src',
    'src.core',
    'src.core.recognizer',
    'src.core.punctuation',
    'src.core.lazy_model_manager',
    'src.core.hotkey_manager',
    'src.core.output_manager',
    'src.core.audio_capture',
    'src.ui',
    'src.ui.main_window',
    'src.ui.tray_icon',
    'src.ui.themes',
    'src.ui.tabs',
    'src.ui.widgets',
    'src.data',
    'src.data.config',
    'src.data.database',
    'src.data.models_manager',
    'src.utils',
    'src.utils.constants',
    'src.utils.logger',
    'src.utils.autostart',
    'src.utils.system_info',
]

# Исключаемые модули (для уменьшения размера)
excludes = [
    # PyTorch - НЕ НУЖЕН для ONNX версии!
    'torch',
    'torch.*',
    'torchaudio',
    'torchvision',

    # Графические библиотеки
    'matplotlib',
    'PIL',
    'cv2',
    'opencv',

    # GUI библиотеки которые не используются
    'tkinter',
    '_tkinter',
    'Tkinter',

    # Тестирование
    'test',
    'tests',
    'pytest',
    'unittest',

    # Разработка
    'IPython',
    'jupyter',
    'notebook',

    # Научные библиотеки (не нужны)
    'scipy',
    'pandas',
    'sklearn',
    'scikit-learn',

    # TensorFlow (не нужен)
    'tensorflow',
    'tensorboard',
    'keras',

    # Примеры numpy
    'numpy.random._examples',
]

a = Analysis(
    [str(ROOT_DIR / 'src' / 'main.py')],
    pathex=[str(ROOT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(ROOT_DIR / 'build' / 'rthook_pynput.py'),  # Pre-initialize pynput for hotkeys
    ],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoiceType',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Без консольного окна
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT_DIR / 'resources' / 'icons' / 'app_icon.ico'),
    version=str(ROOT_DIR / 'build' / 'version_info.txt') if (ROOT_DIR / 'build' / 'version_info.txt').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Не сжимаем DLL которые могут сломаться от UPX
        'vcruntime140.dll',
        'python*.dll',
        'Qt*.dll',
        # Vosk DLLs
        'libvosk.dll',
        'libgcc*.dll',
        'libstdc++*.dll',
        'libwinpthread*.dll',
    ],
    name='VoiceType',
)
