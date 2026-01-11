#!/usr/bin/env python
"""
VoiceType Build Script
Сборка приложения для production.

Использование:
    python build.py              # Полная сборка
    python build.py --clean      # Только очистка
    python build.py --no-clean   # Сборка без очистки
    python build.py --help       # Помощь

Требования:
    - PyInstaller: pip install pyinstaller
    - Pillow (для иконки): pip install pillow
    - Inno Setup (для установщика): https://jrsoftware.org/isdl.php
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Корневая директория проекта
ROOT_DIR = Path(__file__).parent.parent
BUILD_DIR = ROOT_DIR / "build"
DIST_DIR = BUILD_DIR / "dist"
SPEC_FILE = BUILD_DIR / "voicetype.spec"
INSTALLER_SCRIPT = BUILD_DIR / "installer.iss"


def print_header(text: str):
    """Печать заголовка."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, total: int, text: str):
    """Печать шага."""
    print(f"\n[{step}/{total}] {text}...")


def clean_build():
    """Очистить артефакты предыдущей сборки."""
    print_step(1, 5, "Cleaning previous build artifacts")

    dirs_to_clean = [
        DIST_DIR,
        BUILD_DIR / "build",
        ROOT_DIR / "__pycache__",
        ROOT_DIR / "src" / "__pycache__",
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"  Removing: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)

    # Удаляем .pyc файлы
    for pyc_file in ROOT_DIR.rglob("*.pyc"):
        pyc_file.unlink(missing_ok=True)

    print("  Clean complete")


def check_requirements():
    """Проверить наличие необходимых инструментов."""
    print_step(2, 5, "Checking requirements")

    errors = []

    # Проверяем PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        errors.append("PyInstaller not found. Install: pip install pyinstaller")

    # Проверяем наличие spec файла
    if SPEC_FILE.exists():
        print(f"  Spec file: {SPEC_FILE.name}")
    else:
        errors.append(f"Spec file not found: {SPEC_FILE}")

    # Проверяем иконку
    icon_path = ROOT_DIR / "resources" / "icons" / "app_icon.ico"
    if icon_path.exists():
        print(f"  App icon: OK")
    else:
        print(f"  App icon: MISSING (will try to create)")
        # Пробуем создать иконку
        try:
            create_icon_script = BUILD_DIR / "create_icon.py"
            if create_icon_script.exists():
                result = subprocess.run(
                    [sys.executable, str(create_icon_script)],
                    cwd=str(BUILD_DIR),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  App icon: Created")
                else:
                    print(f"  Warning: Could not create icon")
        except Exception as e:
            print(f"  Warning: Could not create icon: {e}")

    if errors:
        print("\nErrors found:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def run_pyinstaller():
    """Запустить PyInstaller."""
    print_step(3, 5, "Running PyInstaller")

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--workpath", str(BUILD_DIR / "build"),
        "--distpath", str(DIST_DIR),
        str(SPEC_FILE)
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        # Показываем вывод в реальном времени
    )

    if result.returncode != 0:
        print(f"\n  ERROR: PyInstaller failed with code {result.returncode}")
        return False

    print("  PyInstaller complete")
    return True


def post_process():
    """Пост-обработка: удаление ненужных файлов."""
    print_step(4, 5, "Post-processing")

    dist_app_dir = DIST_DIR / "VoiceType"

    if not dist_app_dir.exists():
        print(f"  Warning: dist directory not found: {dist_app_dir}")
        return False

    # Удаляем .lib файлы (не нужны для runtime)
    lib_files = list(dist_app_dir.rglob("*.lib"))
    if lib_files:
        print(f"  Removing {len(lib_files)} .lib files...")
        for lib_file in lib_files:
            lib_file.unlink(missing_ok=True)

    # Удаляем .pdb файлы (debug symbols)
    pdb_files = list(dist_app_dir.rglob("*.pdb"))
    if pdb_files:
        print(f"  Removing {len(pdb_files)} .pdb files...")
        for pdb_file in pdb_files:
            pdb_file.unlink(missing_ok=True)

    # Подсчитываем размер
    total_size = sum(f.stat().st_size for f in dist_app_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)

    print(f"  Final size: {size_mb:.1f} MB")
    print("  Post-processing complete")

    return True


def print_summary():
    """Печать итоговой информации."""
    print_step(5, 5, "Build summary")

    dist_app_dir = DIST_DIR / "VoiceType"
    exe_path = dist_app_dir / "VoiceType.exe"

    print(f"\n  Build directory: {dist_app_dir}")

    if exe_path.exists():
        exe_size = exe_path.stat().st_size / (1024 * 1024)
        print(f"  Executable: {exe_path.name} ({exe_size:.1f} MB)")
        print(f"\n  SUCCESS: Build completed!")
    else:
        print(f"\n  ERROR: Executable not found!")
        return False

    # Информация об установщике
    if INSTALLER_SCRIPT.exists():
        print(f"\n  To create installer:")
        print(f"    1. Install Inno Setup: https://jrsoftware.org/isdl.php")
        print(f"    2. Open: {INSTALLER_SCRIPT}")
        print(f"    3. Build -> Compile (Ctrl+F9)")
        print(f"\n  Or run from command line:")
        print(f'    "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe" "{INSTALLER_SCRIPT}"')

    return True


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="VoiceType Build Script")
    parser.add_argument("--clean", action="store_true", help="Only clean, don't build")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning step")
    args = parser.parse_args()

    start_time = datetime.now()

    print_header("VoiceType Build Script")
    print(f"  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {ROOT_DIR}")

    # Только очистка
    if args.clean:
        clean_build()
        print("\nClean complete!")
        return 0

    # Очистка (если не отключена)
    if not args.no_clean:
        clean_build()

    # Проверка требований
    if not check_requirements():
        print("\nBuild failed: requirements not met")
        return 1

    # PyInstaller
    if not run_pyinstaller():
        print("\nBuild failed: PyInstaller error")
        return 1

    # Пост-обработка
    post_process()

    # Итог
    if not print_summary():
        return 1

    # Время сборки
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n  Build time: {duration.total_seconds():.1f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
