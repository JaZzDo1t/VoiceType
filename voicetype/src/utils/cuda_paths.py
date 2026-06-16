"""
Утилита добавления путей к CUDA DLL на Windows.

Используется WhisperRecognizer и download_model.py.
"""
import os
import sys

from loguru import logger

NVIDIA_PACKAGES = ['cublas', 'cudnn', 'cufft', 'cuda_runtime', 'nvjitlink']


def add_cuda_dll_paths() -> None:
    """
    Добавить пути к CUDA DLL для Windows.

    Нужно вызывать перед каждой загрузкой модели, т.к. после unload()
    DLL могут стать недоступны.

    Использует два метода для совместимости:
    1. os.add_dll_directory() - официальный способ для Python 3.8+
    2. PATH environment variable - для библиотек использующих LoadLibrary напрямую
    """
    if os.name != 'nt':  # Только для Windows
        return

    try:
        # sys.prefix указывает на venv когда он активирован
        # Это надёжнее чем искать в sys.path
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

        if not os.path.exists(site_packages):
            logger.warning(f"Не найден site-packages: {site_packages}")
            return

        logger.debug(f"Используем site-packages: {site_packages}")

        # Пути к nvidia DLL (проверяем разные структуры пакетов)
        nvidia_paths = []
        nvidia_base = os.path.join(site_packages, 'nvidia')
        if os.path.exists(nvidia_base):
            for pkg in NVIDIA_PACKAGES:
                # Проверяем оба варианта: bin и lib/x64
                for subdir in ['bin', os.path.join('lib', 'x64'), 'lib']:
                    path = os.path.join(nvidia_base, pkg, subdir)
                    if os.path.exists(path):
                        nvidia_paths.append(path)

        added_count = 0
        current_path = os.environ.get('PATH', '')
        path_additions = []

        for path in nvidia_paths:
            try:
                # Метод 1: os.add_dll_directory (Python 3.8+)
                os.add_dll_directory(path)
                added_count += 1
                logger.debug(f"add_dll_directory: {path}")
            except Exception as e:
                logger.debug(f"add_dll_directory failed для {path}: {e}")

            # Метод 2: добавляем в PATH для LoadLibrary
            if path not in current_path:
                path_additions.append(path)

        # Обновляем PATH
        if path_additions:
            new_path = ';'.join(path_additions) + ';' + current_path
            os.environ['PATH'] = new_path
            logger.debug(f"Добавлено в PATH: {len(path_additions)} путей")

        if added_count > 0:
            logger.info(f"Добавлено {added_count} путей к CUDA DLL")
        else:
            logger.warning("Не найдены пути к CUDA DLL в nvidia packages")

    except Exception as e:
        logger.warning(f"Ошибка добавления CUDA DLL путей: {e}")
