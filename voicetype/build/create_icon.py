#!/usr/bin/env python
"""
VoiceType - Icon Generator
Создает иконку приложения app_icon.ico

Требуется: pip install pillow

Использование:
    python create_icon.py
"""
import sys
from pathlib import Path

# Добавляем корень проекта в path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow не установлен!")
    print("Установите: pip install pillow")
    sys.exit(1)


def create_icon_image(size: int) -> Image.Image:
    """
    Создать изображение иконки заданного размера.

    Args:
        size: Размер в пикселях (квадрат)

    Returns:
        PIL Image объект
    """
    # Цвета
    bg_color = (64, 64, 64)  # Темно-серый фон
    text_color = (255, 255, 255)  # Белый текст
    accent_color = (76, 175, 80)  # Зеленый акцент (Material Green 500)

    # Создаем изображение с прозрачным фоном
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Рисуем скругленный прямоугольник (фон)
    margin = int(size * 0.05)
    radius = int(size * 0.15)

    # Полный фон
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=bg_color
    )

    # Микрофон (акцент) - маленький индикатор
    mic_size = int(size * 0.15)
    mic_x = size - margin - mic_size - int(size * 0.08)
    mic_y = margin + int(size * 0.08)
    draw.ellipse(
        [mic_x, mic_y, mic_x + mic_size, mic_y + mic_size],
        fill=accent_color
    )

    # Текст "VT"
    text = "VT"

    # Размер шрифта
    font_size = int(size * 0.45)

    # Пробуем использовать системный шрифт
    font = None
    font_names = [
        "arial.ttf",
        "Arial.ttf",
        "calibri.ttf",
        "Calibri.ttf",
        "segoeui.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        # Используем встроенный шрифт (меньше качество)
        font = ImageFont.load_default()

    # Получаем размеры текста
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Позиция текста (центр)
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - int(size * 0.05)  # Немного выше центра

    # Рисуем текст
    draw.text((x, y), text, font=font, fill=text_color)

    return img


def create_ico_file(output_path: Path, sizes: list = None) -> bool:
    """
    Создать .ico файл с несколькими размерами.

    Args:
        output_path: Путь к выходному файлу
        sizes: Список размеров (по умолчанию [16, 32, 48, 256])

    Returns:
        True если успешно
    """
    if sizes is None:
        sizes = [16, 32, 48, 256]

    images = []

    for size in sizes:
        img = create_icon_image(size)
        images.append(img)
        print(f"  Created {size}x{size} icon")

    # Сохраняем как .ico
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Главное изображение - самое большое
    main_img = images[-1]

    # Сохраняем с несколькими размерами
    main_img.save(
        str(output_path),
        format='ICO',
        sizes=[(s, s) for s in sizes]
    )

    print(f"\nIcon saved to: {output_path}")
    return True


def create_png_icons(output_dir: Path, sizes: list = None) -> bool:
    """
    Создать PNG иконки разных размеров.

    Args:
        output_dir: Директория для сохранения
        sizes: Список размеров

    Returns:
        True если успешно
    """
    if sizes is None:
        sizes = [16, 32, 48, 64, 128, 256]

    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        img = create_icon_image(size)
        output_path = output_dir / f"app_icon_{size}.png"
        img.save(str(output_path), format='PNG')
        print(f"  Saved {output_path.name}")

    return True


def main():
    """Основная функция."""
    print("=" * 50)
    print("VoiceType Icon Generator")
    print("=" * 50)

    # Пути
    icons_dir = ROOT_DIR / "resources" / "icons"
    ico_path = icons_dir / "app_icon.ico"

    print(f"\nOutput directory: {icons_dir}")
    print(f"ICO file: {ico_path}")

    # Создаем .ico файл
    print("\n[1/2] Creating ICO file...")
    sizes = [16, 32, 48, 256]

    try:
        create_ico_file(ico_path, sizes)
    except Exception as e:
        print(f"Error creating ICO: {e}")
        return 1

    # Опционально: создаем PNG иконки
    print("\n[2/2] Creating PNG icons...")
    try:
        create_png_icons(icons_dir, [16, 32, 48, 64, 128, 256])
    except Exception as e:
        print(f"Error creating PNGs: {e}")
        # Не критично, продолжаем

    print("\n" + "=" * 50)
    print("Icon generation complete!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
