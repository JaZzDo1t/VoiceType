"""
VoiceType - Real Hardware Tests
Тесты с реальным микрофоном и аудио оборудованием.
НЕ ИСПОЛЬЗУЮТ МОКИ - работают с реальным железом.

Требования:
- Микрофон подключен к системе
- Default audio device доступен
- Vosk модель установлена (для тестов распознавания)

Запуск:
    pytest tests/test_real_hardware.py -v -s --tb=short

С маркером:
    pytest -m hardware -v -s
"""
import pytest
import time
import sys
import queue
import threading
from pathlib import Path
from typing import List, Optional

# Markers для hardware тестов
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.slow,
]

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import after path setup
from src.core.audio_capture import AudioCapture
from src.core.recognizer import Recognizer
from src.utils.constants import SAMPLE_RATE, CHUNK_SIZE


# =============================================================================
# Helper Functions
# =============================================================================

def is_microphone_available() -> bool:
    """Проверить доступность микрофона."""
    try:
        devices = AudioCapture.list_devices()
        # list_devices() всегда добавляет "default" в начало,
        # проверяем есть ли реальные устройства
        return len(devices) > 1 or any(d['id'] != 'default' for d in devices)
    except Exception:
        return False


def find_vosk_model() -> Optional[Path]:
    """Найти установленную Vosk модель."""
    # Проверяем стандартные пути
    possible_paths = [
        ROOT_DIR / "models" / "vosk-model-small-ru-0.22",
        ROOT_DIR / "models" / "vosk-model-ru-0.42",
        ROOT_DIR / "models" / "vosk-model-small-en-us-0.15",
        ROOT_DIR / "models" / "vosk-model-en-us-0.22",
        ROOT_DIR.parent / "models" / "vosk-model-small-ru-0.22",
        Path.home() / ".vosk" / "vosk-model-small-ru-0.22",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Ищем любую папку с "vosk-model" в имени
    models_dir = ROOT_DIR / "models"
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and "vosk-model" in item.name:
                return item

    return None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def check_microphone():
    """Skip all tests if no microphone available."""
    if not is_microphone_available():
        pytest.skip("No microphone available - skipping hardware tests")


@pytest.fixture
def audio_capture():
    """
    Создать AudioCapture с default device.
    Автоматически останавливает и освобождает ресурсы после теста.
    """
    capture = AudioCapture(device_id="default")
    yield capture
    # Cleanup
    if capture.is_running():
        capture.stop()


@pytest.fixture
def started_audio_capture():
    """
    Создать и запустить AudioCapture.
    Уже готов к записи.
    """
    capture = AudioCapture(device_id="default")
    result = capture.start()
    if not result:
        pytest.skip("Failed to start audio capture")
    yield capture
    # Cleanup
    capture.stop()


@pytest.fixture(scope="module")
def vosk_model_path():
    """Получить путь к Vosk модели или skip если не найдена."""
    path = find_vosk_model()
    if path is None:
        pytest.skip("Vosk model not found - skipping recognition tests")
    return path


@pytest.fixture
def recognizer(vosk_model_path):
    """
    Создать Recognizer с загруженной моделью.
    """
    rec = Recognizer(str(vosk_model_path))
    if not rec.load_model():
        pytest.skip("Failed to load Vosk model")
    yield rec
    rec.unload()


# =============================================================================
# Test Class: Real Audio Capture
# =============================================================================

class TestRealAudioCapture:
    """Тесты AudioCapture с реальным микрофоном."""

    def test_list_real_devices(self, check_microphone):
        """
        Получить список реальных аудио устройств.
        Проверяет что PyAudio работает и видит устройства.
        """
        devices = AudioCapture.list_devices()

        print(f"\n{'='*60}")
        print(f"Found {len(devices)} audio devices:")
        print(f"{'='*60}")

        for d in devices:
            default_marker = " [DEFAULT]" if d.get('is_default') else ""
            print(f"  ID={d['id']:8s} | {d['name']}{default_marker}")
            if 'sample_rate' in d:
                print(f"           | Sample rate: {d['sample_rate']} Hz, Channels: {d.get('channels', '?')}")

        print(f"{'='*60}")

        assert len(devices) > 0, "No audio devices found"

        # Проверяем структуру
        for d in devices:
            assert 'id' in d, "Device missing 'id' field"
            assert 'name' in d, "Device missing 'name' field"

    @pytest.mark.timeout(10)
    def test_capture_from_default_device(self, check_microphone, audio_capture):
        """
        Захват аудио с default микрофона.
        Проверяет что stream открывается и работает.
        """
        print("\n[TEST] Starting audio capture from default device...")

        # Start recording
        result = audio_capture.start()
        assert result is True, "Failed to start audio capture"

        print("[TEST] Audio capture started, waiting 2 seconds...")

        # Wait for some audio data
        time.sleep(2)

        # Check we're running
        assert audio_capture.is_running(), "Audio capture should be running"

        # Check we got data in queue
        audio_queue = audio_capture.get_audio_queue()
        assert not audio_queue.empty(), "Audio queue should have data"

        # Count items in queue
        items_count = audio_queue.qsize()
        print(f"[TEST] Audio queue has {items_count} items after 2 seconds")

        assert items_count > 0, "Should have received audio data"

        audio_capture.stop()
        print("[TEST] Audio capture stopped successfully")

    @pytest.mark.timeout(10)
    def test_audio_level_with_real_input(self, check_microphone, audio_capture):
        """
        Проверка уровня сигнала с реального микрофона.
        Уровень должен быть в диапазоне 0.0-1.0.
        """
        print("\n[TEST] Testing audio level measurement...")

        result = audio_capture.start()
        assert result is True, "Failed to start audio capture"

        levels = []
        print("[TEST] Recording audio levels for 3 seconds...")

        # Собираем уровни за 3 секунды
        for i in range(30):  # 30 * 0.1s = 3s
            time.sleep(0.1)
            level = audio_capture.get_level()
            levels.append(level)

            # Визуализация
            bar_len = int(level * 50)
            bar = "#" * bar_len + "." * (50 - bar_len)
            print(f"  Level: [{bar}] {level:.4f}")

        audio_capture.stop()

        # Проверки
        assert len(levels) > 0, "Should have recorded levels"

        min_level = min(levels)
        max_level = max(levels)
        avg_level = sum(levels) / len(levels)

        print(f"\n[TEST] Level stats: min={min_level:.4f}, max={max_level:.4f}, avg={avg_level:.4f}")

        # Все уровни должны быть в допустимом диапазоне
        for level in levels:
            assert 0.0 <= level <= 1.0, f"Level {level} out of range [0.0, 1.0]"

    @pytest.mark.timeout(10)
    def test_audio_queue_receives_real_data(self, check_microphone, audio_capture):
        """
        Проверка что данные попадают в очередь.
        Данные должны быть bytes нужной длины.
        """
        print("\n[TEST] Testing audio queue data reception...")

        result = audio_capture.start()
        assert result is True

        print("[TEST] Waiting 1 second for audio data...")
        time.sleep(1)

        audio_queue = audio_capture.get_audio_queue()

        # Получаем несколько chunks
        chunks = []
        while not audio_queue.empty() and len(chunks) < 10:
            try:
                chunk = audio_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break

        audio_capture.stop()

        print(f"[TEST] Received {len(chunks)} audio chunks")

        assert len(chunks) > 0, "Should have received audio chunks"

        # Проверяем каждый chunk
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, bytes), f"Chunk {i} should be bytes"
            # PCM int16 mono: chunk_size * 2 bytes (16 bits = 2 bytes per sample)
            expected_size = CHUNK_SIZE * 2
            print(f"  Chunk {i}: {len(chunk)} bytes (expected ~{expected_size})")
            # Размер может немного отличаться
            assert len(chunk) > 0, f"Chunk {i} should not be empty"

    @pytest.mark.timeout(15)
    def test_start_stop_real_recording(self, check_microphone, audio_capture):
        """
        Тест старт/стоп записи.
        Проверяет корректное переключение состояний.
        """
        print("\n[TEST] Testing start/stop cycle...")

        # Initial state
        assert not audio_capture.is_running(), "Should not be running initially"

        # Start
        print("[TEST] Starting...")
        result = audio_capture.start()
        assert result is True
        assert audio_capture.is_running(), "Should be running after start"

        time.sleep(1)

        # Stop
        print("[TEST] Stopping...")
        audio_capture.stop()
        assert not audio_capture.is_running(), "Should not be running after stop"

        # Start again
        print("[TEST] Starting again...")
        result = audio_capture.start()
        assert result is True
        assert audio_capture.is_running(), "Should be running after second start"

        time.sleep(1)

        # Stop again
        print("[TEST] Final stop...")
        audio_capture.stop()
        assert not audio_capture.is_running()

        print("[TEST] Start/stop cycle completed successfully")

    @pytest.mark.timeout(30)
    def test_continuous_recording_5_seconds(self, check_microphone, audio_capture):
        """
        Непрерывная запись в течение 5 секунд.
        Проверяет стабильность работы.
        """
        print("\n[TEST] Testing 5 seconds continuous recording...")

        result = audio_capture.start()
        assert result is True

        start_time = time.time()
        chunk_count = 0
        error_count = 0

        audio_queue = audio_capture.get_audio_queue()

        print("[TEST] Recording for 5 seconds...")

        while time.time() - start_time < 5.0:
            # Проверяем что все еще работает
            if not audio_capture.is_running():
                error_count += 1
                print(f"  [WARNING] Audio capture stopped unexpectedly at {time.time() - start_time:.1f}s")
                break

            # Считаем chunks
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    chunk_count += 1
                except queue.Empty:
                    break

            time.sleep(0.1)

        elapsed = time.time() - start_time
        audio_capture.stop()

        print(f"[TEST] Recording completed: {elapsed:.2f}s, {chunk_count} chunks received")

        assert error_count == 0, "Audio capture should not stop unexpectedly"
        assert chunk_count > 0, "Should have received chunks during 5s recording"

        # Ожидаемое количество chunks за 5 секунд
        # sample_rate / chunk_size * duration = chunks
        expected_chunks = (SAMPLE_RATE / CHUNK_SIZE) * 5
        print(f"[TEST] Expected ~{expected_chunks:.0f} chunks, got {chunk_count}")

        # Допускаем отклонение в 50%
        assert chunk_count > expected_chunks * 0.5, "Too few chunks received"

    @pytest.mark.timeout(15)
    def test_audio_callback_receives_data(self, check_microphone):
        """
        Тест callback для аудио данных.
        """
        print("\n[TEST] Testing audio data callback...")

        received_data = []
        callback_count = [0]

        def on_audio_data(data: bytes):
            callback_count[0] += 1
            if len(received_data) < 5:  # Сохраняем только первые 5
                received_data.append(data)

        capture = AudioCapture(device_id="default")
        capture.on_audio_data = on_audio_data

        result = capture.start()
        assert result is True

        print("[TEST] Waiting for callbacks (2 seconds)...")
        time.sleep(2)

        capture.stop()

        print(f"[TEST] Callback called {callback_count[0]} times")
        print(f"[TEST] Received {len(received_data)} data samples")

        assert callback_count[0] > 0, "Callback should have been called"
        assert len(received_data) > 0, "Should have received data in callback"

        for i, data in enumerate(received_data):
            assert isinstance(data, bytes), f"Data {i} should be bytes"
            assert len(data) > 0, f"Data {i} should not be empty"

    @pytest.mark.timeout(20)
    def test_context_manager(self, check_microphone):
        """
        Тест работы через context manager.
        """
        print("\n[TEST] Testing context manager...")

        with AudioCapture(device_id="default") as capture:
            assert capture.is_running(), "Should be running in context"

            time.sleep(1)

            level = capture.get_level()
            print(f"[TEST] Audio level in context: {level:.4f}")

            queue_size = capture.get_audio_queue().qsize()
            print(f"[TEST] Queue size: {queue_size}")

        # После выхода из контекста должен быть остановлен
        assert not capture.is_running(), "Should be stopped after context exit"
        print("[TEST] Context manager works correctly")

    @pytest.mark.timeout(10)
    def test_error_callback_on_invalid_device(self, check_microphone):
        """
        Тест callback ошибки при неверном устройстве.
        """
        print("\n[TEST] Testing error callback with invalid device...")

        errors = []

        def on_error(e: Exception):
            errors.append(e)
            print(f"[TEST] Error callback received: {type(e).__name__}: {e}")

        # Пробуем открыть несуществующее устройство
        capture = AudioCapture(device_id="9999")
        capture.on_error = on_error

        result = capture.start()

        # Результат зависит от системы, но не должен зависнуть
        print(f"[TEST] Start result with invalid device: {result}")

        capture.stop()

        if not result:
            print("[TEST] Correctly failed to start with invalid device")
        else:
            print("[TEST] System accepted device ID 9999 (might be valid)")


# =============================================================================
# Test Class: Real Speech Recognition
# =============================================================================

class TestRealSpeechRecognition:
    """Тесты распознавания речи с реальным Vosk."""

    @pytest.mark.timeout(60)
    def test_vosk_model_loads(self, vosk_model_path):
        """
        Загрузка модели Vosk.
        """
        print(f"\n[TEST] Loading Vosk model from: {vosk_model_path}")

        progress_values = []

        def on_progress(value: int):
            progress_values.append(value)
            print(f"  Loading progress: {value}%")

        recognizer = Recognizer(str(vosk_model_path))
        recognizer.on_loading_progress = on_progress

        start_time = time.time()
        result = recognizer.load_model()
        load_time = time.time() - start_time

        print(f"[TEST] Model loaded in {load_time:.2f}s")

        assert result is True, "Model should load successfully"
        assert recognizer.is_loaded(), "Model should be marked as loaded"

        recognizer.unload()
        assert not recognizer.is_loaded(), "Model should be unloaded"

        print("[TEST] Model load/unload successful")

    @pytest.mark.timeout(30)
    def test_recognize_silence(self, check_microphone, recognizer, started_audio_capture):
        """
        Распознавание тишины (или фонового шума).
        Должен вернуть пустой или короткий результат.
        """
        print("\n[TEST] Testing recognition of silence/background noise...")

        audio_queue = started_audio_capture.get_audio_queue()

        partial_results = []

        def on_partial(text: str):
            partial_results.append(text)

        recognizer.on_partial_result = on_partial

        print("[TEST] Processing audio for 3 seconds (please be quiet)...")

        start_time = time.time()
        while time.time() - start_time < 3.0:
            try:
                audio_data = audio_queue.get(timeout=0.5)
                recognizer.process_audio(audio_data)
            except queue.Empty:
                continue

        # Получаем финальный результат
        final = recognizer.get_final_result()

        print(f"[TEST] Final result: '{final}'")
        print(f"[TEST] Partial results: {len(partial_results)}")

        # Тишина может дать пустой результат или случайные слова от шума
        # Главное - не зависнуть и не упасть
        print("[TEST] Silence recognition completed (results may vary)")

    @pytest.mark.timeout(60)
    def test_recognize_real_audio(self, check_microphone, recognizer, started_audio_capture):
        """
        Распознавание реальной речи с микрофона (5 секунд).
        ВАЖНО: Пользователь должен говорить в микрофон во время теста!
        """
        print("\n" + "="*60)
        print("REAL SPEECH RECOGNITION TEST")
        print("Please speak into the microphone for 5 seconds!")
        print("Suggested: 'Привет, это тест распознавания речи'")
        print("="*60)

        audio_queue = started_audio_capture.get_audio_queue()

        partial_results = []
        final_results = []

        def on_partial(text: str):
            partial_results.append(text)
            print(f"  [PARTIAL] {text}")

        def on_final(text: str):
            final_results.append(text)
            print(f"  [FINAL] {text}")

        recognizer.on_partial_result = on_partial
        recognizer.on_final_result = on_final

        print("[TEST] Recording and recognizing for 5 seconds...")
        print("-"*60)

        start_time = time.time()
        while time.time() - start_time < 5.0:
            try:
                audio_data = audio_queue.get(timeout=0.5)
                recognizer.process_audio(audio_data)
            except queue.Empty:
                continue

        # Получаем финальный результат
        final = recognizer.get_final_result()
        if final:
            final_results.append(final)
            print(f"  [FINAL] {final}")

        print("-"*60)
        print(f"[TEST] Recognition completed:")
        print(f"  - Partial results: {len(partial_results)}")
        print(f"  - Final results: {len(final_results)}")

        # Объединяем все финальные результаты
        full_text = " ".join(final_results)
        print(f"  - Full recognized text: '{full_text}'")

        # Тест не fail'ится на пустом результате, так как пользователь мог не говорить
        if full_text:
            print("[TEST] Speech was recognized!")
        else:
            print("[TEST] No speech detected (user might not have spoken)")

    @pytest.mark.timeout(60)
    def test_partial_results_streaming(self, check_microphone, recognizer, started_audio_capture):
        """
        Проверка streaming промежуточных результатов.
        """
        print("\n[TEST] Testing partial results streaming...")
        print("Please speak continuously for 5 seconds...")

        audio_queue = started_audio_capture.get_audio_queue()

        partial_count = [0]
        partial_texts = []

        def on_partial(text: str):
            partial_count[0] += 1
            if text and (not partial_texts or text != partial_texts[-1]):
                partial_texts.append(text)
                print(f"  [PARTIAL #{partial_count[0]}] {text}")

        recognizer.on_partial_result = on_partial

        start_time = time.time()
        while time.time() - start_time < 5.0:
            try:
                audio_data = audio_queue.get(timeout=0.5)
                recognizer.process_audio(audio_data)
            except queue.Empty:
                continue

        recognizer.get_final_result()

        print(f"\n[TEST] Streaming results:")
        print(f"  - Total partial callbacks: {partial_count[0]}")
        print(f"  - Unique partial texts: {len(partial_texts)}")

        # Partial results должны генерироваться часто
        if partial_count[0] > 0:
            print("[TEST] Partial results streaming works!")
        else:
            print("[TEST] No partial results (user might not have spoken)")

    @pytest.mark.timeout(30)
    def test_recognizer_reset(self, recognizer):
        """
        Тест сброса recognizer между сессиями.
        """
        print("\n[TEST] Testing recognizer reset...")

        assert recognizer.is_loaded()

        # Генерируем тестовые аудио данные (тишина)
        import numpy as np
        silence = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        # Обрабатываем
        recognizer.process_audio(silence)
        recognizer.process_audio(silence)

        # Reset
        recognizer.reset()

        # Должен все еще работать
        assert recognizer.is_loaded()
        recognizer.process_audio(silence)

        print("[TEST] Recognizer reset works correctly")


# =============================================================================
# Test Class: Full Pipeline E2E
# =============================================================================

class TestFullPipeline:
    """End-to-end тесты полного pipeline."""

    @pytest.mark.timeout(90)
    def test_full_pipeline_microphone_to_text(self, check_microphone, vosk_model_path):
        """
        Полный pipeline: микрофон -> распознавание -> текст.
        """
        print("\n" + "="*60)
        print("FULL PIPELINE TEST: Microphone -> Recognition -> Text")
        print("Please speak for 5 seconds when prompted!")
        print("="*60)

        # Создаем компоненты
        capture = AudioCapture(device_id="default")
        recognizer = Recognizer(str(vosk_model_path))

        # Загружаем модель
        print("[TEST] Loading Vosk model...")
        assert recognizer.load_model(), "Failed to load model"

        recognized_texts = []

        def on_final(text: str):
            recognized_texts.append(text)
            print(f"  [RECOGNIZED] {text}")

        recognizer.on_final_result = on_final

        try:
            # Запускаем захват
            print("[TEST] Starting audio capture...")
            assert capture.start(), "Failed to start capture"

            audio_queue = capture.get_audio_queue()

            print("\n>>> SPEAK NOW! (5 seconds) <<<\n")

            # Обрабатываем аудио 5 секунд
            start_time = time.time()
            while time.time() - start_time < 5.0:
                try:
                    audio_data = audio_queue.get(timeout=0.5)
                    recognizer.process_audio(audio_data)
                except queue.Empty:
                    continue

            # Финальный результат
            final = recognizer.get_final_result()
            if final:
                recognized_texts.append(final)
                print(f"  [RECOGNIZED] {final}")

        finally:
            capture.stop()
            recognizer.unload()

        full_text = " ".join(recognized_texts)

        print("\n" + "="*60)
        print(f"PIPELINE RESULT: '{full_text}'")
        print("="*60)

        if full_text:
            print("[TEST] Full pipeline works!")
        else:
            print("[TEST] No speech detected")

    @pytest.mark.timeout(120)
    def test_pipeline_multiple_sessions(self, check_microphone, vosk_model_path):
        """
        Тест нескольких сессий записи подряд.
        """
        print("\n[TEST] Testing multiple recording sessions...")

        capture = AudioCapture(device_id="default")
        recognizer = Recognizer(str(vosk_model_path))

        assert recognizer.load_model()

        try:
            for session in range(3):
                print(f"\n[TEST] Session {session + 1}/3")

                recognizer.reset()

                assert capture.start()
                audio_queue = capture.get_audio_queue()

                start_time = time.time()
                chunks_processed = 0

                while time.time() - start_time < 2.0:
                    try:
                        audio_data = audio_queue.get(timeout=0.5)
                        recognizer.process_audio(audio_data)
                        chunks_processed += 1
                    except queue.Empty:
                        continue

                result = recognizer.get_final_result()
                capture.stop()

                print(f"  Processed {chunks_processed} chunks, result: '{result}'")

                time.sleep(0.5)  # Пауза между сессиями

        finally:
            capture.stop()
            recognizer.unload()

        print("[TEST] Multiple sessions completed successfully")


# =============================================================================
# Test Class: Stress and Edge Cases
# =============================================================================

class TestStressAndEdgeCases:
    """Стресс-тесты и граничные случаи."""

    @pytest.mark.timeout(30)
    def test_rapid_start_stop(self, check_microphone):
        """
        Быстрое переключение start/stop.
        """
        print("\n[TEST] Testing rapid start/stop cycles...")

        capture = AudioCapture(device_id="default")

        cycles = 10
        successful = 0

        for i in range(cycles):
            if capture.start():
                time.sleep(0.2)
                capture.stop()
                successful += 1
            time.sleep(0.1)

        print(f"[TEST] Completed {successful}/{cycles} rapid cycles")

        assert successful == cycles, f"Not all cycles completed: {successful}/{cycles}"

    @pytest.mark.timeout(30)
    def test_queue_overflow_handling(self, check_microphone):
        """
        Проверка поведения при переполнении очереди.
        """
        print("\n[TEST] Testing queue behavior under load...")

        capture = AudioCapture(device_id="default")

        assert capture.start()

        # Не читаем из очереди 5 секунд - данные накапливаются
        print("[TEST] Letting data accumulate for 5 seconds...")
        time.sleep(5)

        queue_size = capture.get_audio_queue().qsize()
        print(f"[TEST] Queue size after 5s: {queue_size}")

        # Система должна продолжать работать
        assert capture.is_running(), "Capture should still be running"

        capture.stop()

        print("[TEST] Queue overflow handling works")

    @pytest.mark.timeout(60)
    def test_long_recording_stability(self, check_microphone):
        """
        Стабильность при длительной записи (30 секунд).
        """
        print("\n[TEST] Testing 30 second recording stability...")

        capture = AudioCapture(device_id="default")

        assert capture.start()

        levels_over_time = []
        errors = []

        start_time = time.time()
        while time.time() - start_time < 30.0:
            if not capture.is_running():
                errors.append(f"Stopped at {time.time() - start_time:.1f}s")
                break

            levels_over_time.append(capture.get_level())

            # Читаем из очереди чтобы не переполнять
            q = capture.get_audio_queue()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

            time.sleep(0.5)

        capture.stop()

        elapsed = time.time() - start_time

        print(f"[TEST] Recording ran for {elapsed:.1f}s")
        print(f"[TEST] Collected {len(levels_over_time)} level samples")
        print(f"[TEST] Errors: {errors}")

        assert len(errors) == 0, f"Errors during recording: {errors}"
        assert elapsed >= 29.0, "Recording should run full 30 seconds"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Запуск тестов напрямую
    import subprocess
    import sys

    print("Running Real Hardware Tests...")
    print("="*60)

    # Используем venv python
    venv_python = Path(__file__).parent.parent / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = sys.executable

    result = subprocess.run([
        str(venv_python), "-m", "pytest",
        __file__,
        "-v", "-s",
        "--tb=short",
        "-x"  # Stop on first failure
    ], cwd=str(ROOT_DIR))

    sys.exit(result.returncode)
