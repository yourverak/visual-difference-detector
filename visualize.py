import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from solution import detect_diff


def get_alignment_debug_data(image1: np.ndarray, image2: np.ndarray):
    """
    Выполняет поиск маркеров и аффинное выравнивание для визуализации.
    Возвращает координаты маркеров и трансформированное изображение.
    """
    H_IMG, W_IMG = 600, 800
    TARGET_H, TARGET_W = 485, 685
    
    is_black = np.sum(image2.astype(np.uint16), axis=2) == 0
    y_coords, x_coords = np.where(is_black)
    
    if len(x_coords) == 0:
        return None, None

    points = np.column_stack((x_coords, y_coords))
    cx, cy = W_IMG // 2, H_IMG // 2
    
    quadrants = {
        'TL': points[(points[:, 0] < cx) & (points[:, 1] < cy)],
        'TR': points[(points[:, 0] >= cx) & (points[:, 1] < cy)],
        'BR': points[(points[:, 0] >= cx) & (points[:, 1] >= cy)],
        'BL': points[(points[:, 0] < cx) & (points[:, 1] >= cy)]
    }
    
    detected_centers = []
    for key in ['TL', 'TR', 'BR', 'BL']:
        pts = quadrants[key]
        centroid = np.mean(pts, axis=0) if len(pts) > 0 else [0, 0]
        detected_centers.append(centroid)
            
    src_pts = np.array(detected_centers)
    
    ref_centers = np.array([[67.5, 67.5], [732.5, 67.5], [732.5, 532.5], [67.5, 532.5]])
    X_dst = np.hstack([ref_centers, np.ones((4, 1))])
    
    affine_matrix, _, _, _ = np.linalg.lstsq(X_dst, src_pts, rcond=None)
    
    crop_offset = 57.5
    grid_y, grid_x = np.indices((TARGET_H, TARGET_W))
    global_dst_x = (grid_x + crop_offset).flatten()
    global_dst_y = (grid_y + crop_offset).flatten()
    dst_coords_mat = np.vstack([global_dst_x, global_dst_y, np.ones_like(global_dst_x)]).T
    
    src_coords_flat = dst_coords_mat @ affine_matrix
    
    x_near = np.clip(np.round(src_coords_flat[:, 0]).astype(int), 0, W_IMG - 1)
    y_near = np.clip(np.round(src_coords_flat[:, 1]).astype(int), 0, H_IMG - 1)
    
    warped_img = image2[y_near, x_near].reshape(TARGET_H, TARGET_W, 3)
    return src_pts, warped_img


def run_visualization_pipeline(samples_dir: str = 'samples', save_plots: bool = True):
    """
    Визуализирует все этапы работы алгоритма и сохраняет результаты 
    с английскими подписями для портфолио.
    """
    image1_files = sorted(glob.glob(os.path.join(samples_dir, '*_image1.png')))
    
    if not image1_files:
        print("[ERROR] No files found. Please check the 'samples/' directory.")
        return

    output_dir = os.path.join('docs', 'images')
    os.makedirs(output_dir, exist_ok=True)

    for path_img1 in image1_files:
        prefix = path_img1.replace('_image1.png', '')
        path_img2 = prefix + '_image2.png'
        test_name = os.path.basename(prefix)
        
        if not os.path.exists(path_img2):
            continue

        print(f"Processing visualization for: {test_name}...")
        
        img1 = np.array(Image.open(path_img1).convert('RGB'))
        img2 = np.array(Image.open(path_img2).convert('RGB'))
        

        final_mask = detect_diff(img1, img2)
        markers, warped_img = get_alignment_debug_data(img1, img2)
        
        if markers is None:
            print(f"[SKIP] Alignment failed for {test_name}")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Visual Difference & Alignment Analysis: {test_name}', fontsize=14, weight='bold')

  
        ax = axes[0, 0]
        ax.imshow(img2)
        ax.set_title("1. Distorted Input & Detected Markers", fontsize=11, pad=10)
        ax.scatter(markers[:, 0], markers[:, 1], c='red', s=100, marker='o', edgecolors='white', linewidth=2)
        
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, txt in enumerate(labels):
            ax.annotate(txt, (markers[i, 0]+12, markers[i, 1]+12), color='yellow', fontsize=10, weight='bold')
        ax.axis('off')

    
        ax = axes[0, 1]
        crop_start = 58
        img1_crop = img1[crop_start:crop_start+485, crop_start:crop_start+685]
        ax.imshow(img1_crop)
        ax.set_title("2. Reference Image (Cropped Target)", fontsize=11, pad=10)
        ax.axis('off')


        ax = axes[1, 0]
        ax.imshow(warped_img)
        ax.set_title("3. Aligned & Warped Image (Bilinear)", fontsize=11, pad=10)
        ax.axis('off')


        ax = axes[1, 1]
        ax.imshow(final_mask, cmap='gray')
        ax.set_title("4. Output Difference Mask (Eroded)", fontsize=11, pad=10)
        ax.axis('off')

        plt.tight_layout()
        
        if save_plots:
            export_path = os.path.join(output_dir, f'{test_name}_pipeline.png')
            plt.savefig(export_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] Exported English visualization to: {export_path}")
            
        plt.show()
        
        ans = input("Proceed to the next test sample? (y/n): ")
        if ans.lower() != 'y':
            break


if __name__ == "__main__":
    run_visualization_pipeline()