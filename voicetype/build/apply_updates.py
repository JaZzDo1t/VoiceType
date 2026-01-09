#!/usr/bin/env python
"""
VoiceType - Apply Build Updates
Применяет обновления файлов для production сборки.

Заменяет:
- src/core/punctuation.py <- punctuation_new.py
- src/data/models_manager.py <- models_manager_new.py
- build/voicetype.spec <- voicetype_new.spec

Использование:
    python apply_updates.py
"""
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def main():
    """Применить обновления."""
    print("=" * 50)
    print("VoiceType - Applying Build Updates")
    print("=" * 50)

    updates = [
        # (source, destination, description)
        (
            ROOT_DIR / "src" / "core" / "punctuation_new.py",
            ROOT_DIR / "src" / "core" / "punctuation.py",
            "Punctuation module (local Silero support)"
        ),
        (
            ROOT_DIR / "src" / "data" / "models_manager_new.py",
            ROOT_DIR / "src" / "data" / "models_manager.py",
            "Models manager (Silero path support)"
        ),
        (
            ROOT_DIR / "build" / "voicetype_new.spec",
            ROOT_DIR / "build" / "voicetype.spec",
            "PyInstaller spec (full configuration)"
        ),
    ]

    success_count = 0
    skip_count = 0

    for source, dest, desc in updates:
        print(f"\n{desc}")
        print(f"  Source: {source.name}")
        print(f"  Dest:   {dest.name}")

        if not source.exists():
            print(f"  SKIP: Source file not found")
            skip_count += 1
            continue

        try:
            # Создаем бэкап если файл существует
            if dest.exists():
                backup = dest.with_suffix(dest.suffix + ".bak")
                shutil.copy2(dest, backup)
                print(f"  Backup: {backup.name}")

            # Копируем новый файл
            shutil.copy2(source, dest)
            print(f"  OK: Updated")
            success_count += 1

            # Удаляем _new файл
            source.unlink()
            print(f"  Cleaned: {source.name} removed")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n" + "=" * 50)
    print(f"Results: {success_count} updated, {skip_count} skipped")
    print("=" * 50)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
