# Анализ рисков перехода на PyTorch CPU-only

**Дата:** 2026-01-09
**Цель:** Выявить потенциальные проблемы ПЕРЕД внедрением

---

## Резюме

| Перспектива | Уровень риска | Вероятность | Митигация |
|-------------|---------------|-------------|-----------|
| **PyInstaller bundling** | СРЕДНИЙ | 40% | Требуется тестирование, возможны проблемы с hiddenimports |
| **TorchScript JIT loading** | НИЗКИЙ | 15% | Код уже использует `map_location='cpu'` |
| **Silero TE совместимость** | НИЗКИЙ | 10% | JIT модель не зависит от CUDA |

**Общий вывод:** Переход ВОЗМОЖЕН, но требует тщательного тестирования PyInstaller сборки.

---

## Перспектива 1: PyInstaller + torch CPU-only

### Что может сломаться

#### 1.1 Неактуальные hiddenimports в voicetype.spec

**Проблема:** Текущий `voicetype.spec` содержит CUDA-специфичные импорты:

```python
# build/voicetype.spec:114-116
'torch.backends.cuda',
'torch.backends.cudnn',
```

**Риск:** В CPU-only версии PyTorch эти модули:
- **Могут отсутствовать** → ошибка сборки
- **Могут быть пустыми заглушками** → молчаливый сбой при импорте

**Симптомы:**
- `ModuleNotFoundError: No module named 'torch.backends.cuda'`
- Молчаливое падение при запуске exe без сообщения об ошибке

**Решение:** Обернуть в try-except или удалить из hiddenimports.

#### 1.2 collect_submodules('torch') соберёт CPU-only структуру

**Текущий код (voicetype.spec:62-64):**
```python
torch_hiddenimports = collect_submodules('torch')
torch_distributed_imports = collect_submodules('torch.distributed')
torch_utils_imports = collect_submodules('torch.utils')
```

**Риск:** При CPU-only установке:
- `collect_submodules` найдёт другой набор модулей
- Могут отсутствовать ожидаемые CUDA-модули
- Runtime hook может не инициализироваться правильно

**Симптомы:**
- exe собирается, но падает при первом использовании torch
- `[rthook_torch] PyTorch init failed: ...`

**Решение:** Явно перечислить только необходимые модули, убрать автосбор.

#### 1.3 Runtime hook (rthook_torch.py) может сломаться

**Текущая логика (build/rthook_torch.py:100-112):**
```python
# Step 7: Verify basic functionality
_ = torch.__version__
_ = torch.nn.Module

# Step 8: Test JIT model loading capability
@torch.jit.script
def _test_fn(x: torch.Tensor) -> torch.Tensor:
    return x + 1
```

**Риск:** Тест JIT scripting может работать по-другому в CPU-only:
- `torch.jit.script` компилирует по-другому без CUDA backend
- Могут быть warning'и о несовместимости

**Симптомы:**
- `sys._torch_rthook_success = False`
- Silero TE не загружается, используется PunctuationDisabled

**Решение:** Протестировать rthook с CPU-only torch перед сборкой.

### Источники по PyInstaller + torch

- [PyInstaller Discussion #6230: torch.jit error](https://github.com/orgs/pyinstaller/discussions/6230)
- [PyInstaller Issue #8348: Unable to import torch in exe](https://github.com/pyinstaller/pyinstaller/issues/8348)
- [PyInstaller Issue #2666: Hook for pytorch](https://github.com/pyinstaller/pyinstaller/issues/2666)

---

## Перспектива 2: TorchScript JIT model loading

### Что может сломаться

#### 2.1 RecursiveScriptModule._construct может отсутствовать

**Проблема:** [PyTorch Issue #150089](https://github.com/pytorch/pytorch/issues/150089) - для frozen моделей `torch.compile` падает с:
```
AttributeError: 'RecursiveScriptModule' object has no attribute 'training'
```

**Текущая защита в коде (rthook_torch.py:38-80):**
```python
if not hasattr(RecursiveScriptModule, '_construct'):
    print("[rthook_torch] RecursiveScriptModule._construct is missing, patching...")
    # Определяет _construct classmethod
    RecursiveScriptModule._construct = _construct
```

**Риск:** Патч может работать по-другому в CPU-only версии:
- Внутренние словари (строки 52-68) могут отличаться
- `_c` (C++ module pointer) может иметь другую структуру

**Вероятность:** НИЗКАЯ - патч касается Python-уровня, не CUDA.

#### 2.2 torch.jit.load с map_location='cpu'

**Текущий код (punctuation.py:207):**
```python
self.model = torch_module.jit.load(str(model_path), map_location='cpu')
```

**Анализ:** Это ПРАВИЛЬНЫЙ подход!
- `map_location='cpu'` явно указывает загружать на CPU
- Не зависит от наличия CUDA
- Работает одинаково в full и CPU-only torch

**Риск:** МИНИМАЛЬНЫЙ - код уже готов к CPU-only.

#### 2.3 Атрибуты не сохраняются после load/save

**Проблема:** [PyTorch Issue #127679](https://github.com/pytorch/pytorch/issues/127679) - externally assigned атрибуты не переживают `torch.jit.save/load`.

**Текущая защита (punctuation.py:240-256):**
```python
def _verify_model(self):
    """Verify that model forward() works."""
    # Create minimal test input
    x = torch.zeros(1, 5, dtype=torch.long)
    # Try calling forward
    with torch.no_grad():
        result = self.model(x, att, lan)
```

**Анализ:** Модель проверяется сразу после загрузки, что хорошо.

### Источники по TorchScript

- [PyTorch Issue #68559: forward attribute missing](https://github.com/pytorch/pytorch/issues/68559)
- [PyTorch torch.jit.load documentation](https://docs.pytorch.org/docs/stable/generated/torch.jit.load.html)

---

## Перспектива 3: Silero TE совместимость

### Что может сломаться

#### 3.1 JIT модель Silero зависит от CUDA?

**Анализ файлов модели:**
```
voicetype/models/silero-te/
├── .gitkeep
├── te_wrapper.py        # Reference implementation
├── te_model_jit.pt      # TorchScript model (нужно скачать)
└── te_tokenizer_jit.pt  # TorchScript tokenizer (нужно скачать)
```

**Текущий код (te_wrapper.py:28):**
```python
model = torch.jit.load(model_path, map_location='cpu')
```

**Анализ:** JIT модель Silero:
- Создана с `torch.jit.script` или `torch.jit.trace`
- Сохранена как платформо-независимый TorchScript
- НЕ содержит CUDA-специфичных операций (только nn.Linear, nn.Embedding и т.д.)

**Риск:** МИНИМАЛЬНЫЙ - модель изначально CPU-совместима.

#### 3.2 torch.hub.load fallback может сломаться

**Текущий код (punctuation.py:694-699):**
```python
# Fallback: загрузка из torch.hub (требует интернет)
self._model, _, _, _, self._apply_te = _torch_module.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_te',
    ...
)
```

**Риск:** torch.hub может загрузить модель с CUDA tensors:
- Silero репозиторий может вернуть CUDA-модель по умолчанию
- `apply_te` функция может ожидать CUDA

**Симптомы:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Вероятность:** НИЗКАЯ - Silero модели официально поддерживают CPU.

**Решение:** В frozen builds torch.hub НЕ используется (только JIT), поэтому это не критично.

#### 3.3 numpy операции в enhance_tokens

**Текущий код (punctuation.py:88-89):**
```python
punct_np = punct.cpu().numpy()
capital_np = capital.cpu().numpy()
```

**Анализ:** Явный вызов `.cpu()` перед `.numpy()` - ПРАВИЛЬНО!
- Работает одинаково в full и CPU-only torch
- Tensors уже на CPU, `.cpu()` - no-op

**Риск:** ОТСУТСТВУЕТ.

### Источники по Silero

- [Silero Models GitHub](https://github.com/snakers4/silero-models)
- [Silero Text Enhancement models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml)

---

## Что ТОЧНО сломается

### 1. requirements.txt с --index-url внутри файла

**Проблема:** `--index-url` внутри requirements.txt работает НЕ ВЕЗДЕ:

```
# ЭТО НЕ РАБОТАЕТ в некоторых версиях pip!
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
```

**Симптомы:**
- pip игнорирует строку, устанавливает обычный torch с CUDA
- Poetry/uv вообще не понимают этот синтаксис

**Решение:** Использовать отдельную команду или pip.conf:

```bash
# Правильный способ
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt  # Без torch
```

### 2. Версия torch+cpu может отличаться

**Проблема:** CPU-only версии выходят с небольшим отставанием:
- torch 2.5.0 (CUDA) вышел раньше torch 2.5.0+cpu
- Может не быть точной версии

**Симптомы:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.5.0+cpu
```

**Решение:** Проверить доступность на https://download.pytorch.org/whl/cpu/

### 3. Размер wheel может удивить

**Ожидание:** CPU-only ~200 MB
**Реальность для Windows:** CPU-only ~250-300 MB

Это всё равно значительно меньше ~900 MB с CUDA, но не ~200 MB.

---

## Чек-лист перед внедрением

### Фаза 1: Локальное тестирование (30 мин)

- [ ] Создать чистое venv
- [ ] `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- [ ] Проверить `python -c "import torch; print(torch.__version__)"`
- [ ] Проверить `python -c "import torch; print(torch.cuda.is_available())"` → должно быть `False`
- [ ] Запустить `python run.py`
- [ ] Протестировать распознавание с пунктуацией

### Фаза 2: Тестирование PyInstaller (1-2 часа)

- [ ] Обновить voicetype.spec:
  - Убрать/закомментировать `torch.backends.cuda`, `torch.backends.cudnn`
  - Проверить collect_submodules собирает нужное
- [ ] Собрать: `pyinstaller build/voicetype.spec`
- [ ] Проверить размер dist/VoiceType/
- [ ] Запустить VoiceType.exe
- [ ] Проверить rthook_torch.py логи в stderr
- [ ] Протестировать распознавание с пунктуацией в exe

### Фаза 3: Документация (15 мин)

- [ ] Обновить README с инструкцией установки CPU-only
- [ ] Документировать fallback на полный torch при проблемах

---

## Рекомендации

### ДЕЛАТЬ:
1. **Тестировать локально ПЕРЕД изменением requirements.txt**
2. **Сначала проверить PyInstaller сборку, потом коммитить**
3. **Оставить возможность отката на полный torch**

### НЕ ДЕЛАТЬ:
1. ~~Добавлять --index-url в requirements.txt~~ - не везде работает
2. ~~Удалять CUDA hiddenimports без тестирования~~ - может сломать сборку
3. ~~Обновлять torch до последней версии одновременно с CPU-only~~ - делать по одному

---

## План отката

Если CPU-only не заработает:

```bash
# Откат на полный torch
pip uninstall torch
pip install torch>=2.0.0,<3.0.0

# Пересборка exe
pyinstaller build/voicetype.spec
```

Время отката: ~15 минут.

---

## Заключение

**Переход на PyTorch CPU-only ВОЗМОЖЕН** при соблюдении условий:

1. ✅ Код punctuation.py уже использует `map_location='cpu'`
2. ✅ JIT модели Silero не зависят от CUDA
3. ⚠️ PyInstaller spec требует адаптации hiddenimports
4. ⚠️ requirements.txt нужно менять осторожно

**Рекомендуемый порядок:**
1. Протестировать локально в venv
2. Протестировать PyInstaller сборку
3. Только после успеха - коммитить изменения

**Оценка рисков:** 30% вероятность проблем с PyInstaller, решаемых за 1-2 часа отладки.
