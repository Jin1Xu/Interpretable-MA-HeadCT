import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# =========================================================================
# 模块导入区域
# =========================================================================

# 1. 获取脚本所在目录
current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)

# 2. 智能判定项目根目录
if os.path.exists(os.path.join(script_dir, "five_class")):
    project_root = script_dir
else:
    project_root = os.path.dirname(script_dir)

# 3. 将根目录加入系统路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. 导入模块
try:
    # --- A. 导入 2D 分析模块 ---
    from explain.data_loader import PngDataset2D
    from explain.config import binary as config_2d
    from explain.model.binary import create_model as create_model_2d

    # --- B. 导入 3D 分类模块 ---
    from five_class.multiclass_3d import BrainCT3DClassifier
    import five_class.multiclass_3d_config as config_3d

except ImportError as e:
    print("="*60)
    print(f"导入失败: {e}")
    print(f"脚本所在目录: {script_dir}")
    print(f"判定根目录为: {project_root}")
    print("请检查 'five_class' 和 'explain' 文件夹是否位于根目录下。")
    print("="*60)
    sys.exit(1)

# =========================================================================
# 配置部分：类别映射与可视化参数
# =========================================================================

CLASS_MAP_3D = {
    "Composite": 0,
    "Acute Massive Cerebral Infarction": 1,
    "Acute Intracerebral Hemorrhage": 2,
    "Acute Subdural/Epidural Hemorrhage": 3,
    "Diffuse Subarachnoid Hemorrhage": 4,
}

PREDICTION_TO_VISUALIZATION = {
    0: 'hemorrhage',       
    1: 'ischemia',         
    2: 'hemorrhage',       
    3: 'extra_sub_dural',  
    4: 'sah'               
}

# =========================================================================
# PART 1: 图像处理与可视化算法
# =========================================================================

def brain_high(img_hu, bg_threshold=0, bone_threshold=150, display_min=0, display_max=80, cmap_name="jet"):
    """ 高密度：脑出血/钙化 """
    h, w = img_hu.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    mask_bone = img_hu > bone_threshold
    mask_bg   = img_hu < bg_threshold
    mask_soft = ~(mask_bone | mask_bg)
    rgba[mask_bone] = [0.9, 0.9, 0.9, 1.0] 
    vals = img_hu[mask_soft]
    vals = np.clip(vals, display_min, display_max)
    norm = (vals - display_min) / (display_max - display_min + 1e-6)
    cmap = plt.get_cmap(cmap_name)
    rgba[mask_soft] = cmap(norm)
    return rgba

def brain_low(img_hu, bg_threshold=0, bone_threshold=150, display_min=0, display_max=90, cmap_name="jet"):
    """ 低密度：脑梗/水肿 """
    h, w = img_hu.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    mask_bone = img_hu > bone_threshold
    mask_bg   = img_hu < bg_threshold
    mask_soft = ~(mask_bone | mask_bg)
    rgba[mask_bone] = [0.85, 0.85, 0.85, 1.0]
    vals = img_hu[mask_soft]
    vals = np.clip(vals, display_min, display_max)
    norm = (vals - display_min) / (display_max - display_min + 1e-6)
    cmap = plt.get_cmap(cmap_name)
    rgba[mask_soft] = cmap(norm)
    return rgba

def brain_extra_sub_dural(img_hu, bg_threshold=0, bone_threshold=100, display_min=0, display_max=80, cmap_name="jet"):
    """ 硬膜外/硬膜下出血 """
    h, w = img_hu.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    mask_bone = img_hu > bone_threshold
    mask_bg   = img_hu < bg_threshold
    mask_soft = ~(mask_bone | mask_bg)
    rgba[mask_bone] = [0.85, 0.85, 0.85, 1.0]
    vals = img_hu[mask_soft]
    vals = np.clip(vals, display_min, display_max)
    norm = (vals - display_min) / (display_max - display_min + 1e-6)
    cmap = plt.get_cmap(cmap_name)
    rgba[mask_soft] = cmap(norm)
    return rgba

def brain_sah(img_hu, bg_threshold=0, bone_threshold=90, display_min=0, display_max=65, cmap_name="jet"):
    """ 蛛网膜下腔出血 """
    h, w = img_hu.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    mask_bone = img_hu > bone_threshold
    mask_bg   = img_hu < bg_threshold
    mask_soft = ~(mask_bone | mask_bg)
    rgba[mask_bone] = [0.85, 0.85, 0.85, 1.0]
    vals = img_hu[mask_soft]
    vals = np.clip(vals, display_min, display_max)
    norm = (vals - display_min) / (display_max - display_min + 1e-6)
    cmap = plt.get_cmap(cmap_name)
    rgba[mask_soft] = cmap(norm)
    return rgba

def brain_midline_shift_numpy(img_hu, pixel_spacing=1.0):
    """ 脑中线偏移可视化 (简化占位) """
    wc, ww = 40, 80
    img_norm = cv2.normalize(np.clip(img_hu, wc-ww//2, wc+ww//2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out_img = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    cv2.putText(out_img, "Midline Shift Mode", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out_img

LESION_CONFIG = {
    'hemorrhage': {'func': brain_high, 'title': 'Intracerebral Hemorrhage', 'desc': 'High Density Region'},
    'ischemia': {'func': brain_low, 'title': 'Cerebral Ischemia/Infarction', 'desc': 'Low Density Region'},
    'extra_sub_dural': {'func': brain_extra_sub_dural, 'title': 'Epidural / Subdural Hemorrhage', 'desc': 'Extra/Sub-dural Hematoma'},
    'sah': {'func': brain_sah, 'title': 'Subarachnoid Hemorrhage', 'desc': 'SAH Visualization'},
    'herniation': {'func': brain_midline_shift_numpy, 'title': 'Brain Herniation / Midline Shift', 'desc': 'Midline Displacement'}
}

# =========================================================================
# PART 2: 3D 分类逻辑
# =========================================================================

def load_3d_classifier(weights_path, device):
    print(f"[Step 1] Loading 3D Classifier from {weights_path}...")
    model = BrainCT3DClassifier(num_classes=config_3d.num_classes, base_channels=config_3d.base_channels)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_3d_volume(nii_img, target_shape=(96, 256, 256)):
    image_data = nii_img.get_fdata().astype(np.float32)
    current_shape = image_data.shape
    if current_shape != target_shape:
        image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(image_tensor, size=target_shape, mode='trilinear', align_corners=False)
        image_data = resized.squeeze().numpy()
    
    min_val, max_val = np.min(image_data), np.max(image_data)
    if max_val == min_val:
        normalized = np.zeros_like(image_data)
    else:
        normalized = (image_data - min_val) / (max_val - min_val)
        normalized = normalized * 2 - 1
    
    return torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

def run_global_diagnosis(model_3d, nii_img, device):
    input_tensor = preprocess_3d_volume(nii_img, target_shape=config_3d.input_shape)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model_3d(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return pred_idx, confidence

# =========================================================================
# PART 3: 2D 切片选择逻辑
# =========================================================================

def load_2d_selector(weights_path, device):
    print(f"[Step 3] Loading 2D Slice Selector from {weights_path}...")
    model = create_model_2d(config_2d)
    state_dict = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    return model

def process_slice_for_2d_model(slice_data, input_shape, window_center=40, window_width=80):
    min_value = window_center - window_width / 2.0
    max_value = window_center + window_width / 2.0
    windowed_img = np.clip(slice_data, min_value, max_value)
    if max_value == min_value: norm_img = windowed_img
    else: norm_img = (windowed_img - min_value) / (max_value - min_value)
    img_uint8 = (norm_img * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_uint8).convert('L')
    image_data = np.array(img_pil).astype(np.float32)
    image_data = PngDataset2D.resize_to_target_2d(image_data, target_shape=input_shape)
    image_data = PngDataset2D.normalize_to_minus_one_one(image_data)
    
    return torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)

def find_critical_slice(model_2d, img_data, device):
    input_shape = getattr(config_2d, 'input_shape', (256, 256))
    total_slices = img_data.shape[2]
    raw_results = []
    print(f"Scanning {total_slices} slices for critical lesion...")
    
    with torch.no_grad():
        for i in range(total_slices):
            slice_raw = img_data[:, :, i]
            input_tensor = process_slice_for_2d_model(slice_raw, input_shape)
            input_tensor = input_tensor.to(device)
            outputs = model_2d(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            raw_results.append({'prob': probs[0, 1].item(), 'index': i})

    scored_results = []
    num_files = len(raw_results)
    for i in range(num_files):
        p_curr = raw_results[i]['prob']
        if num_files == 1: score = p_curr
        elif i == 0: score = (p_curr + raw_results[i+1]['prob']) / 2.0
        elif i == num_files - 1: score = (raw_results[i-1]['prob'] + p_curr) / 2.0
        else: score = (raw_results[i-1]['prob'] + p_curr + raw_results[i+1]['prob']) / 3.0
        scored_results.append({'index': i, 'score': score})

    scored_results.sort(key=lambda x: x['score'], reverse=True)
    best_idx, best_score = scored_results[0]['index'], scored_results[0]['score']
    print(f"Selected Critical Slice: #{best_idx} (Confidence: {best_score:.4f})")
    
    best_slice_hu = np.rot90(img_data[:, :, best_idx], k=0) 
    return best_slice_hu, best_idx, best_score

# =========================================================================
# PART 4: 绘图与保存 (修改版：纯文本输出)
# =========================================================================

def save_comparison_vector(orig_hu, proc_img, lesion_type, slice_idx, score, diagnosis_text, output_path):
    """保存左右对比的综合报告图"""
    info = LESION_CONFIG.get(lesion_type)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 左图：原始脑窗
    wc, ww = 40, 80
    orig_disp = np.clip(orig_hu, wc - ww/2, wc + ww/2)
    axes[0].imshow(orig_disp, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f"Original Slice #{slice_idx}\nAI Diagnosis: {diagnosis_text}", fontsize=12, fontweight='bold')
    
    # 右图：处理结果
    axes[1].imshow(proc_img)
    axes[1].axis('off')
    title_str = f"{info['title']}\n{info['desc']}" if info else "Unknown Lesion Visualization"
    axes[1].set_title(title_str, fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    if not output_path.endswith('.svg'):
        output_path = os.path.splitext(output_path)[0] + '.svg'
        
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
    print(f"Combined Report saved to: {output_path}")
    plt.close()

def save_separate_results(orig_hu, proc_img, diagnosis_text, base_output_path):
    """
    单独保存三个关键变量：
    1. 预测结果文本 (diagnosis_text) -> .txt
    2. 原始切片 SVG (orig_hu) -> _original.svg
    3. 处理后切片 SVG (proc_img) -> _processed.svg
    """
    base_name = os.path.splitext(base_output_path)[0]

    # 1. 保存预测结果为 TXT (仅包含类别名称)
    txt_path = f"{base_name}_prediction.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(diagnosis_text)
    print(f"Prediction Text saved to: {txt_path}")

    # 2. 保存原始切片为 SVG (带脑窗)
    orig_path = f"{base_name}_original.svg"
    plt.figure(figsize=(6, 6))
    wc, ww = 40, 80
    orig_disp = np.clip(orig_hu, wc - ww/2, wc + ww/2)
    plt.imshow(orig_disp, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(orig_path, bbox_inches='tight', pad_inches=0, format='svg')
    plt.close()
    print(f"Original Slice saved to: {orig_path}")

    # 3. 保存处理后切片为 SVG
    proc_path = f"{base_name}_processed.svg"
    plt.figure(figsize=(6, 6))
    plt.imshow(proc_img)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(proc_path, bbox_inches='tight', pad_inches=0, format='svg')
    plt.close()
    print(f"Processed Slice saved to: {proc_path}")

# =========================================================================
# 主程序
# =========================================================================

if __name__ == "__main__":
    
    # ---------------------------------------------------------------------
    # 手动修改的区域
    # ---------------------------------------------------------------------
    
    DEFAULT_MODEL_3D = os.path.join(project_root, "five_class", "best_model.pth")
    DEFAULT_MODEL_2D = os.path.join(project_root, "explain", "weight", "best_model.pth")
    DEFAULT_INPUT_NII = "test_datasets/脑出血/00079557_CT00-B_UID_4cf4f2_S2_DESC_5mm_Stnd_4717.nii"
    DEFAULT_OUTPUT = "analysis_result.svg"

    # ---------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(description="End-to-End Brain CT Lesion Diagnosis & Analysis")
    parser.add_argument('--input_nii', type=str, default=DEFAULT_INPUT_NII)
    parser.add_argument('--model_3d_path', type=str, default=DEFAULT_MODEL_3D)
    parser.add_argument('--model_2d_path', type=str, default=DEFAULT_MODEL_2D)
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--force_lesion_type', type=str, default=None, choices=list(LESION_CONFIG.keys()))

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isabs(args.input_nii):
        potential_path = os.path.join(project_root, args.input_nii)
        if os.path.exists(potential_path): args.input_nii = potential_path
            
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(project_root, args.output_path)

    print("="*60)
    print(f"Running Analysis in: {project_root}")
    print(f"Input File  : {args.input_nii}")
    print(f"Output Base : {args.output_path}")
    print("="*60)

    try:
        if not os.path.exists(args.input_nii):
            raise FileNotFoundError(f"Input file not found: {args.input_nii}")
        
        nii = nib.load(args.input_nii)
        img_data_raw = nii.get_fdata() 
        try: pixel_spacing = float(nii.header.get_zooms()[0])
        except: pixel_spacing = 1.0

        # Step 1: 3D Diagnosis
        if args.force_lesion_type:
            lesion_type = args.force_lesion_type
            diagnosis_str = f"Manual Override: {lesion_type}"
            print(f"Skipping 3D classification, using forced type: {lesion_type}")
        else:
            model_3d = load_3d_classifier(args.model_3d_path, device)
            pred_idx, conf = run_global_diagnosis(model_3d, nii, device)
            idx_to_name = {v: k for k, v in CLASS_MAP_3D.items()}
            class_name_cn = idx_to_name.get(pred_idx, "Unknown")
            lesion_type = PREDICTION_TO_VISUALIZATION.get(pred_idx, 'hemorrhage')
            
            diagnosis_str = class_name_cn
            
            print(f"Diagnosis Result: {diagnosis_str} (Confidence: {conf:.1%})")
            print(f"Mapped to Visualization Mode: {lesion_type}")
            del model_3d
            torch.cuda.empty_cache()

        # Step 2: 2D Localization
        model_2d = load_2d_selector(args.model_2d_path, device)
        best_slice_hu, slice_idx, score = find_critical_slice(model_2d, img_data_raw, device)
        del model_2d
        torch.cuda.empty_cache()

        # Step 3: Visualization
        print(f"[Step 4] Generating visualization for '{lesion_type}'...")
        vis_func = LESION_CONFIG[lesion_type]['func']
        if lesion_type == 'herniation':
            processed_img = vis_func(best_slice_hu, pixel_spacing=pixel_spacing)
        else:
            processed_img = vis_func(best_slice_hu)
        
        # 保存对比总图
        save_comparison_vector(best_slice_hu, processed_img, lesion_type, slice_idx, score, diagnosis_str, args.output_path)
        
        # 保存独立结果
        save_separate_results(best_slice_hu, processed_img, diagnosis_str, args.output_path)
        
        print("\nPipeline completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()