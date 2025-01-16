import cv2
import numpy as np
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# -----------------------------
# Параметры и диапазоны цветов
# -----------------------------
# HSV диапазоны для красного (с двумя диапазонами)
RED_LOWER1 = np.array([0, 100, 100])  # Увеличены значения насыщенности и яркости
RED_UPPER1 = np.array([8, 255, 255])  # Сужен диапазон оттенков
RED_LOWER2 = np.array([165, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# Диапазоны для желтого и зеленого
YELLOW_LOWER = np.array([22, 130, 130])  # Увеличены значения для более точного определения
YELLOW_UPPER = np.array([32, 255, 255])
GREEN_LOWER = np.array([45, 100, 100])  # Сдвинут диапазон для лучшего определения
GREEN_UPPER = np.array([80, 255, 255])

# Базовый коэффициент для минимальной площади (от общего числа пикселей)
DEFAULT_THRESHOLD_FACTOR = 0.001  # Увеличен порог

# Множители для корректировки порогов для каждого цвета
THRESHOLD_MULTIPLIER_RED = 1.0
THRESHOLD_MULTIPLIER_YELLOW = 1.2
THRESHOLD_MULTIPLIER_GREEN = 0.9

# Ядро для морфологических операций
KERNEL = np.ones((5, 5), np.uint8)  # Увеличено ядро для лучшей фильтрации шума

# -----------------------------
# Функции логирования и утилиты
# -----------------------------
def setup_logging() -> logging.Logger:
    """Настраивает логирование и возвращает объект логгера."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directories(base_path: Path) -> Path:
    """Создаёт (или очищает) папки для результатов классификации."""
    if base_path.exists():
        shutil.rmtree(base_path)
    for color in ['red', 'yellow', 'green', 'unknown']:
        (base_path / color).mkdir(parents=True, exist_ok=True)
    return base_path

def write_classification_results(classifications: Dict[str, List[str]]):
    """Записывает результаты классификации по каждому цвету в отдельный текстовый файл (кроме 'unknown')."""
    for color, filenames in classifications.items():
        if color != 'unknown' and filenames:
            output_file = Path(f'{color}.txt')
            try:
                output_file.write_text('\n'.join(sorted(filenames)), encoding='utf-8')
            except Exception as e:
                print(f"Ошибка записи в файл {output_file}: {e}")

# -----------------------------
# Функции для динамического порогирования и обработки масок
# -----------------------------
def adjust_thresholds(min_threshold: float, avg_value: float, avg_saturation: float):
    """
    Корректирует минимальный порог в зависимости от средней яркости и насыщенности изображения.
    """
    brightness_factor = 1.0
    if avg_value < 90:
        brightness_factor = 0.7
    elif avg_value > 160:
        brightness_factor = 1.3

    saturation_factor = 1.0
    if avg_saturation < 50:
        saturation_factor = 0.8
    elif avg_saturation > 150:
        saturation_factor = 1.2

    final_factor = (brightness_factor + saturation_factor) / 2

    return (min_threshold * THRESHOLD_MULTIPLIER_RED * final_factor,
            min_threshold * THRESHOLD_MULTIPLIER_YELLOW * final_factor,
            min_threshold * THRESHOLD_MULTIPLIER_GREEN * final_factor)

def analyze_contours(mask, min_area=20):
    """
    Анализирует контуры на маске с улучшенной фильтрацией.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_area = 0
    count = 0
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # Проверяем форму контура
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Анализируем выпуклость
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Улучшенные критерии фильтрации
        if circularity > 0.4 and solidity > 0.8:
            valid_area += area
            count += 1
            
    return valid_area, count

# -----------------------------
# Функция выделения региона интереса (ROI)
# -----------------------------
def extract_roi(img):
    """
    Улучшенное выделение ROI с адаптивными границами.
    """
    h, w = img.shape[:2]
    
    # Адаптивное определение границ в зависимости от соотношения сторон
    aspect_ratio = w / h
    
    if aspect_ratio > 1.5:  # Широкое изображение
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)
        y1 = int(h * 0.1)
        y2 = int(h * 0.9)
    else:  # Более квадратное изображение
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * 0.15)
        y2 = int(h * 0.85)
        
    return img[y1:y2, x1:x2]

# -----------------------------
# Функции классификации цвета светофора
# -----------------------------
def detect_traffic_light_color(image_path, logger):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 'unknown'
        
        # Базовая предобработка
        height, width = img.shape[:2]
        if height > 300:
            scale = 300 / height
            width = int(width * scale)
            height = 300
            img = cv2.resize(img, (width, height))
        
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        # Расширяем диапазон для зеленого
        red_lower1 = np.array([0, 90, 90])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 90, 90])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 120, 120])
        yellow_upper = np.array([34, 255, 255])
        
        green_lower = np.array([35, 80, 80])  # Расширили диапазон и снизили пороги
        green_upper = np.array([90, 255, 255])
        
        # Создаем маски
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.add(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Морфология
        kernel = np.ones((3,3), np.uint8)
        for mask in [red_mask, yellow_mask, green_mask]:
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
        
        def analyze_contours(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 20:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.4:  # Немного снизили требование к округлости
                    valid_areas.append(area)
            return sum(valid_areas), len(valid_areas)
        
        # Анализ цветов
        red_pixels, red_count = analyze_contours(red_mask)
        yellow_pixels, yellow_count = analyze_contours(yellow_mask)
        green_pixels, green_count = analyze_contours(green_mask)
        
        min_threshold = int(height * width * 0.0007)  # Немного снизили базовый порог
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        # Проверка на несколько сигналов
        signals = [
            (red_pixels, 'red'),
            (yellow_pixels, 'yellow'),
            (green_pixels, 'green')
        ]
        signals.sort(reverse=True)
        
        # Если два сильных сигнала примерно равны
        if signals[1][0] > signals[0][0] * 0.8:
            return 'unknown'
        
        # Классификация с измененными порогами
        if max_pixels < min_threshold:
            return 'unknown'
        elif max_pixels == yellow_pixels and yellow_pixels > min_threshold * 1.3:
            return 'yellow'
        elif max_pixels == red_pixels and red_pixels > min_threshold * 1.1:
            return 'red'
        elif max_pixels == green_pixels and green_pixels > min_threshold * 0.9:  # Снизили порог для зеленого
            return 'green'
        else:
            return 'unknown'
            
    except Exception:
        return 'unknown'

# -----------------------------
# Основная логика обработки изображений
# -----------------------------
def main():
    logger = setup_logging()
    logger.info("Начало классификации изображений светофоров")
    
    input_path = Path('tlights')
    output_path = Path('classified_images')
    
    if not input_path.exists():
        logger.error(f"Входная директория {input_path} не существует")
        return
    
    create_directories(output_path)
    
    classifications: Dict[str, List[str]] = {
        'red': [],
        'yellow': [],
        'green': [],
        'unknown': []
    }
    
    images = list(input_path.glob('*.jpg'))
    if not images:
        logger.warning(f"В директории {input_path} не найдено файлов с расширением *.jpg")
    
    for image_path in tqdm(images, desc="Классификация изображений"):
        color = detect_traffic_light_color(image_path, logger)
        classifications[color].append(image_path.name)
        try:
            shutil.copy2(image_path, output_path / color / image_path.name)
        except Exception as e:
            logger.exception(f"Ошибка копирования {image_path} в {output_path / color}: {e}")
    
    write_classification_results(classifications)
    
    total_images = len(images)
    logger.info("Классификация завершена")
    logger.info("Статистика:")
    for color, filenames in classifications.items():
        count = len(filenames)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        logger.info(f"{color}: {count} изображений ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
