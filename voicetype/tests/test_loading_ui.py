"""
VoiceType - Loading UI Tests
Тесты для экрана загрузки и состояния загрузки моделей.
"""
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.main_window import MainWindow
from src.ui.widgets.loading_overlay import LoadingOverlay


class TestLoadingOverlay:
    """Тесты виджета загрузки."""

    @pytest.fixture
    def overlay(self, qtbot):
        """Создать виджет загрузки."""
        widget = LoadingOverlay()
        qtbot.addWidget(widget)
        return widget

    def test_initial_state(self, overlay):
        """Проверить начальное состояние."""
        assert overlay._base_status == "Загрузка"
        assert overlay._model_name == ""

    def test_set_status(self, overlay):
        """Проверить установку статуса."""
        overlay.set_status("Загрузка Vosk", "vosk-model-ru-0.42")

        assert overlay._base_status == "Загрузка Vosk"
        assert overlay._model_name == "vosk-model-ru-0.42"
        assert "vosk-model-ru-0.42" in overlay._model_label.text()

    def test_set_status_without_model(self, overlay):
        """Проверить установку статуса без имени модели."""
        overlay.set_status("Инициализация")

        assert overlay._base_status == "Инициализация"
        assert overlay._model_label.text() == ""

    def test_show_error(self, overlay):
        """Проверить отображение ошибки."""
        overlay.show_error("Ошибка загрузки")

        assert "Ошибка загрузки" in overlay._status_label.text()
        # Анимация должна остановиться
        assert not overlay._animation_timer.isActive()

    def test_reset(self, overlay):
        """Проверить сброс состояния."""
        overlay.show_error("Ошибка")
        overlay.reset()

        assert overlay._base_status == "Загрузка"
        assert overlay._animation_timer.isActive()

    def test_animation_stops_on_hide(self, overlay, qtbot):
        """Проверить остановку анимации при скрытии."""
        overlay.show()
        assert overlay._animation_timer.isActive()

        overlay.hide()
        assert not overlay._animation_timer.isActive()


class TestMainWindowLoading:
    """Тесты состояния загрузки главного окна."""

    @pytest.fixture
    def main_window(self, qtbot):
        """Создать главное окно."""
        window = MainWindow()
        qtbot.addWidget(window)
        return window

    def test_initial_loading_state(self, main_window):
        """Окно начинается в состоянии загрузки."""
        assert main_window._is_loading is True
        assert main_window._stacked_widget.currentIndex() == 0  # Loading overlay

    def test_set_loading_false_switches_to_tabs(self, main_window):
        """set_loading(False) переключает на вкладки."""
        main_window.set_loading(False, "test-model")

        assert main_window._is_loading is False
        assert main_window._stacked_widget.currentIndex() == 1  # Tabs
        assert "Настройки" in main_window.windowTitle()

    def test_set_loading_true_switches_to_overlay(self, main_window):
        """set_loading(True) переключает на экран загрузки."""
        # Сначала переключаемся на вкладки
        main_window.set_loading(False)

        # Теперь обратно на загрузку
        main_window.set_loading(True, "new-model")

        assert main_window._is_loading is True
        assert main_window._stacked_widget.currentIndex() == 0
        assert "Загрузка" in main_window.windowTitle()

    def test_set_loading_updates_tab_test(self, main_window):
        """set_loading(False) обновляет статус в tab_test."""
        main_window.set_loading(False, "vosk-model-small-ru")

        assert main_window.tab_test._models_ready is True
        assert "vosk-model-small-ru" in main_window.tab_test._status_label.text()

    def test_set_loading_status(self, main_window):
        """set_loading_status обновляет статус в overlay."""
        main_window.set_loading_status("Загрузка Silero", "silero-te")

        assert main_window._loading_overlay._base_status == "Загрузка Silero"

    def test_show_loading_error(self, main_window):
        """show_loading_error показывает ошибку."""
        main_window.show_loading_error("Модель не найдена")

        assert "Ошибка" in main_window.windowTitle()

    def test_is_loading_property(self, main_window):
        """Проверить is_loading()."""
        assert main_window.is_loading() is True

        main_window.set_loading(False)
        assert main_window.is_loading() is False


class TestTabTestModelsReady:
    """Тесты готовности моделей в tab_test."""

    @pytest.fixture
    def main_window(self, qtbot):
        """Создать главное окно."""
        window = MainWindow()
        qtbot.addWidget(window)
        return window

    def test_initial_button_disabled(self, main_window):
        """Кнопка теста изначально неактивна."""
        assert main_window.tab_test._test_btn.isEnabled() is False

    def test_set_models_ready_enables_button(self, main_window):
        """set_models_ready(True) активирует кнопку."""
        main_window.tab_test.set_models_ready(True, "test-model")

        assert main_window.tab_test._test_btn.isEnabled() is True
        assert main_window.tab_test._models_ready is True

    def test_set_models_ready_false_disables_button(self, main_window):
        """set_models_ready(False) деактивирует кнопку."""
        main_window.tab_test.set_models_ready(True)
        main_window.tab_test.set_models_ready(False)

        assert main_window.tab_test._test_btn.isEnabled() is False
        assert "Загрузка" in main_window.tab_test._status_label.text()

    def test_model_name_in_status(self, main_window):
        """Имя модели отображается в статусе."""
        main_window.tab_test.set_models_ready(True, "vosk-model-ru-0.42")

        assert "vosk-model-ru-0.42" in main_window.tab_test._status_label.text()
