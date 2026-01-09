"""
VoiceType - Entry Point
Точка входа приложения.
"""
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def main():
    """Главная функция запуска приложения."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon

    from src.utils.logger import setup_logger
    from src.utils.constants import APP_NAME, APP_DATA_DIR
    from src.app import VoiceTypeApp

    # Настраиваем логирование
    setup_logger()

    from loguru import logger
    logger.info(f"Starting {APP_NAME}...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"App data dir: {APP_DATA_DIR}")

    # Создаём директорию данных
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Создаём Qt приложение
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)  # Не закрываться при закрытии окна

    # Устанавливаем иконку приложения
    icon_path = ROOT_DIR / "resources" / "icons" / "app_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Проверяем, что приложение не запущено дважды
    # (простая проверка через lock-файл)
    lock_file = APP_DATA_DIR / "voicetype.lock"

    if lock_file.exists():
        try:
            # Проверяем, жив ли процесс
            with open(lock_file, "r") as f:
                pid = int(f.read().strip())

            # Проверяем существование процесса
            import psutil
            if psutil.pid_exists(pid):
                logger.warning(f"VoiceType already running (PID: {pid})")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    None,
                    APP_NAME,
                    f"{APP_NAME} уже запущен.\n\nПроверьте иконку в системном трее."
                )
                return 1
        except (ValueError, FileNotFoundError, PermissionError):
            pass

    # Создаём lock-файл
    try:
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"Failed to create lock file: {e}")

    # Создаём и инициализируем приложение
    voice_app = VoiceTypeApp()

    if not voice_app.initialize():
        logger.error("Failed to initialize application")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            APP_NAME,
            "Не удалось инициализировать приложение.\n\n"
            "Проверьте логи для деталей."
        )
        return 1

    # Уведомление покажется после загрузки модели
    logger.info(f"{APP_NAME} started successfully")

    # Запускаем event loop
    try:
        exit_code = app.exec()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        exit_code = 1
    finally:
        # Удаляем lock-файл
        try:
            lock_file.unlink()
        except Exception:
            pass

    logger.info(f"{APP_NAME} exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
