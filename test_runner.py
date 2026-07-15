import os
import glob
import time
import numpy as np
from PIL import Image


from solution import detect_diff


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Вычисляет метрику Intersection over Union (IoU) между предсказанной
    и эталонной бинарными масками.
    """
    pred_bool = pred_mask == 255
    true_bool = true_mask == 255
    
    intersection = np.logical_and(pred_bool, true_bool).sum()
    union = np.logical_or(pred_bool, true_bool).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def run_test_suite(samples_dir: str = 'samples'):
    """
    Сканирует директорию с тестовыми образцами и проводит
    полный цикл тестирования алгоритма выравнивания.
    """
    if not os.path.exists(samples_dir):
        print(f"[ERROR] Directory '{samples_dir}' not found. Please place test samples here.")
        return

    image1_files = sorted(glob.glob(os.path.join(samples_dir, '*_image1.png')))
    if not image1_files:
        print(f"[ERROR] No test cases starting with '*_image1.png' found in '{samples_dir}'.")
        return

    total_iou = 0.0
    valid_tests = 0
    
    print(f"\n{'TEST NAME':<20} | {'TIME (s)':<10} | {'IoU':<10} | {'STATUS':<10}")
    print("-" * 60)

    for path_img1 in image1_files:
        prefix = path_img1.replace('_image1.png', '')
        path_img2 = prefix + '_image2.png'
        path_mask = prefix + '_true_mask.png'
        test_name = os.path.basename(prefix)

        if not (os.path.exists(path_img2) and os.path.exists(path_mask)):
            print(f"{test_name:<20} | {'MISSING CORE FILES':<38}")
            continue

        try:
            # Загрузка и приведение к нужным форматам
            img1 = np.array(Image.open(path_img1).convert('RGB'))
            img2 = np.array(Image.open(path_img2).convert('RGB'))
            true_mask = np.array(Image.open(path_mask).convert('L'))

            # Замер времени выполнения
            start_time = time.time()
            pred_mask = detect_diff(img1, img2)
            elapsed = time.time() - start_time

            # Проверка соответствия выходного разрешения контракту задачи
            if pred_mask.shape != (485, 685):
                print(f"{test_name:<20} | {elapsed:.4f}     | {'ERROR':<10} | INVALID SHAPE {pred_mask.shape}")
                continue

            iou = compute_iou(pred_mask, true_mask)
            status = "TIMEOUT" if elapsed > 30.0 else "OK"
            
            if status == "TIMEOUT":
                iou = 0.0

            print(f"{test_name:<20} | {elapsed:.4f}     | {iou:.4f}     | {status}")
            
            total_iou += iou
            valid_tests += 1

        except Exception as err:
            print(f"{test_name:<20} | CRITICAL ERROR: {str(err)}")

    print("-" * 60)
    avg_iou = total_iou / valid_tests if valid_tests > 0 else 0.0
    print(f"Mean IoU: {avg_iou:.4f}")
    print(f"Total Score: {int(avg_iou * 100)} / 100\n")


if __name__ == "__main__":
    run_test_suite()