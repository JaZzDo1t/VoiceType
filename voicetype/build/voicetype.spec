# -*- mode: python ; coding: utf-8 -*-
"""
VoiceType - PyInstaller Specification
Конфигурация для сборки приложения в exe.

Вариант: Полный с PyTorch (~450 MB)
- Полноценная пунктуация Silero
- Полностью офлайн работа
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
                # Кладём DLLs в папку vosk, чтобы модуль их нашёл
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

    # Модели Silero TE (пунктуация)
    (str(ROOT_DIR / 'models' / 'silero-te'), 'models/silero-te'),
]

# Фильтруем несуществующие пути
datas = [(src, dst) for src, dst in datas if Path(src).exists()]

# КРИТИЧНО: Собираем .py файлы для torch.jit (нужны для RecursiveScriptModule._construct)
# Без этого torch.package не может загружать модели в frozen build
datas += collect_data_files('torch.jit', include_py_files=True)
datas += collect_data_files('torch.package', include_py_files=True)

# Собираем все субмодули torch для корректной работы

# Собираем все субмодули torch и torch.distributed
torch_hiddenimports = collect_submodules('torch')
torch_distributed_imports = collect_submodules('torch.distributed')
torch_utils_imports = collect_submodules('torch.utils')

# Скрытые импорты
hiddenimports = [
    # Vosk
    'vosk',

    # PyTorch - все субмодули собраны автоматически через collect_submodules
    # Ключевые модули для Silero TE:
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.jit',
    'torch.package',
    'torch.package.package_importer',
    'torch.package._package_unpickler',
    'torch.package._importlib',
    'torch.package._mangling',
    'torch.package._directory_reader',
    'torch.package.file_structure_representation',
    'torch.package.importer',
    'torch.package.glob_group',
    'torch.serialization',
    'torch._C',
    'torch._utils',
    'torch._utils_internal',
    'torch._strobelight',
    'torch._strobelight.compile_time_profiler',
    'torch._vendor',
    'torch._vendor.packaging',
    'torch._vendor.packaging.version',
    'torch.torch_version',
    'torch.version',

    # КРИТИЧЕСКИЕ модули для PyInstaller (часто отсутствуют)
    'torch.autograd',
    'torch.autograd.function',
    'torch.autograd.grad_mode',
    'torch.nn.modules',
    'torch.nn.modules.module',
    'torch.nn.modules.container',
    'torch.nn.modules.linear',
    'torch.nn.modules.activation',
    'torch.jit',
    'torch.jit._script',
    'torch.jit._state',
    'torch.jit._fuser',
    'torch.jit._recursive',
    'torch.jit._serialization',
    'torch.jit.frontend',
    'torch.backends',
    'torch.backends.cuda',
    'torch.backends.cudnn',
    'torch.multiprocessing',
    'torch._utils._element_size_impl',

    # torch.distributed, testing и utils (нужны для загрузки моделей)
    'torch.distributed',
    'torch.distributed.constants',
    'torch.distributed.distributed_c10d',
    'torch.testing',
    'torch.testing._internal',
    'torch.utils',
    'torch.utils.data',
    'torch.utils.data.dataloader',
    'torch.utils.data.dataset',
    'torch.utils.data.sampler',
    'torch.utils._python_dispatch',

    # Зависимости PyTorch
    'typing_extensions',
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

    # Стандартные библиотеки
    'queue',
    'threading',
    'json',
    'sqlite3',
    'wave',
    'struct',

    # Наши модули
    'src',
    'src.core',
    'src.core.recognizer',
    'src.core.punctuation',
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
] + torch_hiddenimports + torch_distributed_imports + torch_utils_imports

# Исключаемые модули (для уменьшения размера)
excludes = [
    # PyTorch - минимум исключений, иначе circular imports!
    'torch._inductor',
    'torch.onnx',
    'torch.utils.tensorboard',

    # Графические библиотеки
    'matplotlib',
    'PIL',
    'cv2',
    'opencv',

    # GUI библиотеки которые не используются
    'tkinter',
    '_tkinter',
    'Tkinter',

    # Тестирование (unittest нужен для torch!)
    'test',
    'tests',
    'pytest',

    # Разработка
    'IPython',
    'jupyter',
    'notebook',

    # Научные библиотеки (не нужны)
    'scipy',
    'pandas',
    'sklearn',
    'scikit-learn',

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
        str(ROOT_DIR / 'build' / 'rthook_torch.py'),   # Pre-initialize torch
        str(ROOT_DIR / 'build' / 'rthook_pynput.py'),  # Pre-initialize pynput for hotkeys
    ],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    # КРИТИЧНО: собираем .py файлы для torch.jit (нужны для RecursiveScriptModule._construct)
    module_collection_mode={
        'torch': 'pyz+py',
        'torch.jit': 'pyz+py',
        'torch.jit._script': 'pyz+py',
        'torch.jit._recursive': 'pyz+py',
        'torch.package': 'pyz+py',
    },
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
