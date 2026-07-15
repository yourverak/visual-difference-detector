import os
import glob
import time
import numpy as np
from PIL import Image


def detect_diff(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:

    H_IMG, W_IMG = 600, 800
    TARGET_H, TARGET_W = 485, 685
    DIFF_THRESHOLD = 50

    # Координаты маркеров на эталоне: TL, TR, BR, BL
    ref_centers = np.array([
        [67.5, 67.5],
        [732.5, 67.5],
        [732.5, 532.5],
        [67.5, 532.5]
    ])

    # 1. Поиск центроидов маркеров 
    is_black = np.sum(image2.astype(np.uint16), axis=2) == 0
    y_coords, x_coords = np.where(is_black)
    
    if len(x_coords) == 0:
        return np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)

    points = np.column_stack((x_coords, y_coords))
    cx, cy = W_IMG // 2, H_IMG // 2
    
    quadrants = [
        points[(points[:, 0] < cx) & (points[:, 1] < cy)],   # TL
        points[(points[:, 0] >= cx) & (points[:, 1] < cy)],  # TR
        points[(points[:, 0] >= cx) & (points[:, 1] >= cy)], # BR
        points[(points[:, 0] < cx) & (points[:, 1] >= cy)]   # BL
    ]
    
    src_pts = np.array([
        np.mean(pts, axis=0) if len(pts) > 0 else [0, 0] 
        for pts in quadrants
    ])

    # 2. Аффинное преобразование (X_dst @ M = Y_src)
    X_dst = np.hstack([ref_centers, np.ones((4, 1))])
    affine_matrix, _, _, _ = np.linalg.lstsq(X_dst, src_pts, rcond=None)

    # 3. Генерация целевой сетки координат и обратный маппинг
    crop_offset_x, crop_offset_y = 57.5, 57.5
    grid_y, grid_x = np.indices((TARGET_H, TARGET_W))
    
    global_dst_x = (grid_x + crop_offset_x).flatten()
    global_dst_y = (grid_y + crop_offset_y).flatten()
    dst_coords_mat = np.vstack([global_dst_x, global_dst_y, np.ones_like(global_dst_x)]).T
    
    src_coords_flat = dst_coords_mat @ affine_matrix
    src_x, src_y = src_coords_flat[:, 0], src_coords_flat[:, 1]

    # 4. Векторизованная билинейная интерполяция
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, W_IMG - 1)
    x1 = np.clip(x0 + 1, 0, W_IMG - 1)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, H_IMG - 1)
    y1 = np.clip(y0 + 1, 0, H_IMG - 1)

    wa = ((x1 - src_x) * (y1 - src_y))[:, np.newaxis]
    wb = ((x1 - src_x) * (src_y - y0))[:, np.newaxis]
    wc = ((src_x - x0) * (y1 - src_y))[:, np.newaxis]
    wd = ((src_x - x0) * (src_y - y0))[:, np.newaxis]

    warped_flat = (
        image2[y0, x0] * wa + 
        image2[y1, x0] * wb + 
        image2[y0, x1] * wc + 
        image2[y1, x1] * wd
    )
    warped_img = warped_flat.reshape(TARGET_H, TARGET_W, 3).astype(np.uint8)

    # 5. Вырезание эталонной области
    start_idx = int(round(crop_offset_x))
    img1_crop = image1[start_idx : start_idx + TARGET_H, start_idx : start_idx + TARGET_W]

    # 6. Евклидова метрика в RGB и фильтрация шума
    diff_norm = np.linalg.norm(
        img1_crop.astype(np.float32) - warped_img.astype(np.float32), 
        axis=2
    )
    mask_noisy = diff_norm > DIFF_THRESHOLD

    # пиксель валиден только при подтверждении 4 соседями
    clean_mask = np.zeros_like(mask_noisy)
    clean_mask[1:-1, 1:-1] = (
        mask_noisy[1:-1, 1:-1] & 
        mask_noisy[0:-2, 1:-1] & 
        mask_noisy[2:, 1:-1] & 
        mask_noisy[1:-1, 0:-2] & 
        mask_noisy[1:-1, 2:]
    )

    return clean_mask.astype(np.uint8) * 255