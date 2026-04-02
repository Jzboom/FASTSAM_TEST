#!/usr/bin/env python3
"""
FastSAM 葡萄分割 - 改进版 v3
改进点：
1. 高对比度可视化：鲜艳固定颜色（红/黄/青/洋红）+ 粗白晕轮廓 + 编号标签
2. 绿叶排除法：排除主体为绿叶的 mask（兼容深蓝/紫/红各色葡萄）
3. 茎部独立路径：细长 + 棕褐色双重验证，避免误标叶柄/栅栏
"""

import os
import sys
import gc
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict

# ──────────────────────── 配置 ───────────────────────────────────────────────
IMAGE_PATH    = "test_2.jpg"
OUTPUT_DIR    = "output"
MODEL_FILE    = "FastSAM-x.pt"
MODEL_URL     = "https://huggingface.co/CASIA-IVA-Lab/FastSAM-x/resolve/main/FastSAM-x.pt"
DEVICE        = "cuda"
CONF          = 0.2        # 稍低，多捕获第三串及茎部
IOU           = 0.6        # 稍低 IoU，减少漏检
IMGSZ         = 1024
# 单 mask 面积过滤
MIN_AREA      = 500        # 最小面积（px²）
MAX_AREA_FRAC = 0.025      # 单个原始 mask 最大占图面积（葡萄粒级别，过滤整串级超大 mask）
# 绿叶排除过滤（适用于各色葡萄：深蓝/紫/红/粉）
# OpenCV HSV H[28-85] + S>50 = 饱和绿叶，超过阈值则丢弃
EXCL_GREEN_H_MIN = 28      # 绿叶 Hue 下限（56° 标准 = 黄绿起点）
EXCL_GREEN_H_MAX = 85      # 绿叶 Hue 上限（170° 标准 = 青绿结束）
EXCL_GREEN_SAT   = 50      # 绿叶最小饱和度
EXCL_GREEN_FRAC  = 0.55    # mask 内 >55% 绿叶像素则排除
# 第一轮膨胀：合并同簇内相邻颗粒（tight）
DILATE_KERNEL = 2          # 极小膨胀核，仅合并物理接触的葡萄粒，降低跨串桥接风险
MIN_BERRY_AR  = 0.20       # 保留的 mask 最小长宽比（过滤细长非葡萄粒 mask）
# 第二轮：按质心距离合并小簇 → 整串葡萄
BUNCH_EPS     = 100        # 质心距离阈值（px），同串葡萄粒聚到一起
# 合并后有效范围过滤
MIN_MERGED_FRAC = 0.005    # 合并后至少 0.5% 图像（过滤极小噪声簇）
MAX_MERGED_FRAC = 0.80     # 合并后不超过 80%（过滤真正全图大块）
# 葡萄常挚在藤糖中上部，排除质心过于偷下的背景 mask
GRAPE_MAX_Y_FRAC = 0.82    # berry mask 质心 Y 坐标超过该比例则为背景丢弃
# 茎部检测（双重验证：极细长 + 棕褐色，防止误检叶柄/栅栏）
STEM_ASPECT     = 0.19     # minAreaRect 短/长比阈值（< MIN_BERRY_AR，消除死区）
STEM_MAX_FRAC   = 0.04     # 茎面积占图像上限（放大允许粗主蔓）
STEM_COLOR_FRAC = 0.15     # 茎至少 15% 像素为棕褐色（H=5-25, S>30）
# 透明度
ALPHA_GRAPE   = 0.70       # 葡萄簇填充透明度
ALPHA_STEM    = 0.88       # 茎部填充透明度

# ── 高对比度调色盘（BGR）── 刻意避开图中紫色/蓝灰/深绿 ──────────────────────
# 鲜红、亮黄、青、洋红、橙、黄绿（BGRf格式）
PALETTE_BGR = [
    (  0,   0, 255),   # 鲜红
    (  0, 230, 230),   # 亮黄
    (230, 230,   0),   # 青
    (200,   0, 200),   # 洋红
    (  0, 140, 255),   # 橙
    ( 60, 230,  60),   # 亮绿（叶片少时可用）
    (255, 100,   0),   # 蓝紫
    (  0, 200, 150),   # 黄绿
]
STEM_BGR = (0, 80, 255)    # 茎部：橙红（BGR）
# ─────────────────────────────────────────────────────────────────────────────

def download_model():
    if Path(MODEL_FILE).exists():
        print(f"[✓] 模型已存在: {MODEL_FILE}")
        return
    print(f"[↓] 正在下载模型: {MODEL_FILE} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as e:
        backup_url = "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt"
        print(f"[!] 主地址失败，备用: {e}")
        urllib.request.urlretrieve(backup_url, MODEL_FILE)
    print("[✓] 下载完成")


def merge_masks(masks_np, dilate_kernel=12, min_area=500, max_area=None,
                min_berry_ar=0.25):
    """
    第一轮：tight 膨胀法合并紧邻 mask（小膨胀核，不跨簇桥接）。
    max_area：过滤单个原始 mask 的面积上限（叶片/背景大片）。
    min_berry_ar：mask 最小长宽比（过滤细长非葡萄 mask）。
    """
    import cv2
    import numpy as np

    H, W = masks_np.shape[1], masks_np.shape[2]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
    )
    combined = np.zeros((H, W), dtype=np.uint8)
    valid_idx = []
    buf = np.empty((H, W), dtype=np.uint8)

    def _aspect_ratio_quick(m):
        pts = np.column_stack(np.where(m))
        if len(pts) < 4:
            return 1.0
        hull = cv2.convexHull(pts[:, ::-1].astype(np.float32))
        _, (w, h), _ = cv2.minAreaRect(hull)
        return (min(w, h) / max(w, h)) if max(w, h) > 0 else 1.0

    for i, m in enumerate(masks_np):
        area = int(m.sum())
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        # 过滤细长mask（非葡萄粒形状）
        if min_berry_ar > 0:
            ar = _aspect_ratio_quick(m)
            if ar < min_berry_ar:
                continue
        buf[:] = m.astype(np.uint8)
        np.bitwise_or(combined, cv2.dilate(buf, kernel), out=combined)
        valid_idx.append(i)

    if not valid_idx:
        return []

    _, label_map = cv2.connectedComponents(combined)

    groups = defaultdict(list)
    for i in valid_idx:
        m = masks_np[i].astype(bool)
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        lbl = label_map[cy, cx]
        if lbl == 0:
            lbls = label_map[m]
            lbls = lbls[lbls > 0]
            if len(lbls) == 0:
                continue
            lbl = int(np.bincount(lbls).argmax())
        groups[lbl].append(i)

    result = []
    for members in groups.values():
        merged = np.zeros((H, W), dtype=bool)
        for idx in members:
            merged |= masks_np[idx].astype(bool)
        result.append(merged)
    return result


def union_find_merge(masks_list, eps):
    """
    第二轮：按质心距离（union-find 单链接）合并小簇 → 整串葡萄。
    eps：质心距离阈值（px）。距离 < eps 的两个簇合并。
    """
    import numpy as np

    n = len(masks_list)
    if n == 0:
        return []

    centroids = []
    for m in masks_list:
        ys, xs = np.where(m)
        centroids.append([float(xs.mean()), float(ys.mean())])
    centroids = np.array(centroids)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centroids[i] - centroids[j]) < eps:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    H, W = masks_list[0].shape
    merged_list = []
    for members in groups.values():
        merged = np.zeros((H, W), dtype=bool)
        for idx in members:
            merged |= masks_list[idx]
        merged_list.append(merged)
    return merged_list


def split_wide_clusters(merged_list, image_width, split_width_frac=0.35):
    """
    对合并后过宽的簇（宽度 > image_width * split_width_frac）沿 X 轴密度谷值分裂。
    适用于两串极近葡萄被合并的情况。
    """
    import numpy as np
    result = []
    for mask in merged_list:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            result.append(mask)
            continue
        width = int(xs.max()) - int(xs.min())
        if width < split_width_frac * image_width:
            result.append(mask)
            continue
        # 找 X 轴分布的密度谷值
        hist, edges = np.histogram(xs, bins=40)
        smooth = np.convolve(hist, np.ones(5) / 5, mode='same')
        # 只考虑中间 1/4~3/4 范围内的谷值，避免边缘
        center_lo = len(smooth) // 4
        center_hi = 3 * len(smooth) // 4
        valley = center_lo + int(np.argmin(smooth[center_lo:center_hi]))
        split_x = int((edges[valley] + edges[valley + 1]) / 2)
        left_mask  = mask.copy(); left_mask[:, split_x:] = False
        right_mask = mask.copy(); right_mask[:, :split_x] = False
        if left_mask.any():
            result.append(left_mask)
        if right_mask.any():
            result.append(right_mask)
    return result


def aspect_ratio(mask_bool):
    """minAreaRect 短/长边之比（越小越细长：0→线段，1→正方形）"""
    import cv2
    import numpy as np
    m_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    pts = np.vstack(contours)
    _, (w, h), _ = cv2.minAreaRect(pts)
    return (min(w, h) / max(w, h)) if max(w, h) > 0 else 1.0


def fill_mask(img_bgr, mask, color_bgr, alpha):
    """原地 alpha 混合填充（BGR canvas）"""
    import numpy as np
    c = np.array(color_bgr, dtype=np.float32)
    region = img_bgr[mask].astype(np.float32)
    img_bgr[mask] = (region * (1 - alpha) + c * alpha).astype(np.uint8)


def draw_halo_contour(img_bgr, mask, color_bgr, thick=3):
    """白色外晕 + 彩色轮廓 —— 在任意背景上均清晰可见"""
    import cv2
    import numpy as np
    m_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours, -1, (255, 255, 255), thick + 4)
    cv2.drawContours(img_bgr, contours, -1, color_bgr, thick)


def draw_label_box(img_bgr, mask, text, color_bgr, font_scale=1.0):
    """在 mask 质心处绘制黑底彩字标签"""
    import cv2
    import numpy as np
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return
    cx, cy = int(xs.mean()), int(ys.mean())
    font, lw = cv2.FONT_HERSHEY_SIMPLEX, 2
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, lw)
    cv2.rectangle(img_bgr,
                  (cx - tw // 2 - 6, cy - th - bl - 6),
                  (cx + tw // 2 + 6, cy + bl + 6),
                  (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (cx - tw // 2, cy),
                font, font_scale, color_bgr, lw, cv2.LINE_AA)


def run_camera():
    """实时相机分割模式（优先使用 Intel RealSense D405，失败时回退到 OpenCV）"""
    import cv2
    import numpy as np
    import subprocess
    import time

    download_model()

    try:
        from ultralytics import FastSAM
    except ImportError:
        print("[✗] ultralytics 未安装，请先运行: pip install ultralytics")
        sys.exit(1)

    # ── 修复1：停止 iio-sensor-proxy 防止屏幕随相机旋转 ────────────────────
    iio_stopped = False
    try:
        r = subprocess.run(
            ["sudo", "systemctl", "stop", "iio-sensor-proxy"],
            capture_output=True, timeout=5
        )
        if r.returncode == 0:
            iio_stopped = True
            print("[✓] 已暂停 iio-sensor-proxy（屏幕将不再随相机旋转）")
        else:
            # 服务不存在或已停止都属于正常
            pass
    except Exception:
        pass
    if not iio_stopped:
        print("[i] 提示：若屏幕随相机旋转，请手动运行:")
        print("        sudo systemctl stop iio-sensor-proxy")

    # ── 连接相机 ──────────────────────────────────────────────────────────────
    use_realsense = False
    pipeline = None
    cap = None

    try:
        import pyrealsense2 as rs

        # 修复2：连接前 hardware_reset，清除上次残留状态
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 设备，请检查 USB 连接")
        for dev in devices:
            print(f"[→] 重置设备: {dev.get_info(rs.camera_info.name)} ...")
            dev.hardware_reset()
        time.sleep(2)   # 等待重枚举完成

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        use_realsense = True
        print("[✓] Intel RealSense D405 已连接（640×480 @ 30fps）")
    except Exception as e:
        print(f"[!] RealSense 初始化失败: {e}")
        print("[→] 回退到 OpenCV VideoCapture（设备 0）...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[✗] 无法打开任何摄像头，请检查连接")
            sys.exit(1)
        print("[✓] OpenCV 摄像头已打开")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"[→] 加载模型 {MODEL_FILE} ...")
    model = FastSAM(MODEL_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame_count = 0
    save_count  = 0

    # ── FPS 计算 ──────────────────────────────────────────────────────────────
    import time
    fps_times = []     # 存储最近的帧时间戳
    fps_window = 30    # 用于计算平均 FPS 的帧数

    # ── 模式状态 ──────────────────────────────────────────────────────────────
    grape_only_mode = False   # 默认全局分割；按 't' 切换为仅葡萄

    print("[→] 开始实时分割")
    print("     按 'q' 退出  |  按 's' 保存当前帧到 output/")
    print("     按 'f' 切换全屏显示  |  按 't' 切换 全局/仅葡萄 模式")

    window_name = "FastSAM Real-time Segmentation (D405)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_start_time = time.time()

            # ── 取帧 ──────────────────────────────────────────────────────────
            if use_realsense:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    print("[!] 帧超时，重试中...")
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("[!] 获取帧失败，退出")
                    break

            frame_count += 1

            # ── FastSAM 推理（实时模式用 640，兼顾速度与精度）────────────────
            results = model(
                frame_bgr,
                device=DEVICE,
                retina_masks=True,
                imgsz=640,
                conf=CONF,
                iou=IOU,
                verbose=False,
            )

            # ── 叠加可视化 ────────────────────────────────────────────────────
            overlay = frame_bgr.copy()
            result  = results[0]
            n_masks = 0

            if result.masks is not None and result.masks.data is not None:
                masks_raw = result.masks.data.bool().cpu().numpy()

                if not grape_only_mode:
                    # ── 全局模式：直接显示所有 mask ──────────────────────────
                    for i, mask in enumerate(masks_raw):
                        color = PALETTE_BGR[i % len(PALETTE_BGR)]
                        fill_mask(overlay, mask, color, 0.5)
                        draw_halo_contour(overlay, mask, color, thick=2)
                    n_masks = len(masks_raw)
                else:
                    # ── 葡萄模式：HSV 颜色 + 形态过滤，保留疑似葡萄的 mask ──
                    H_fr, W_fr = frame_bgr.shape[:2]
                    total_px   = H_fr * W_fr
                    frame_hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                    h_ch = frame_hsv[..., 0]
                    s_ch = frame_hsv[..., 1]

                    berry_max = int(total_px * MAX_AREA_FRAC)
                    grape_masks = []

                    for m in masks_raw:
                        area = int(m.sum())
                        # 面积过滤
                        if area < MIN_AREA or area > berry_max:
                            continue
                        # 形态过滤（排除细长物体）
                        ar = aspect_ratio(m)
                        if ar < MIN_BERRY_AR:
                            continue
                        # 颜色过滤：排除以绿叶为主的 mask
                        h_vals = h_ch[m]
                        s_vals = s_ch[m]
                        green_px = ((h_vals >= EXCL_GREEN_H_MIN) &
                                    (h_vals <= EXCL_GREEN_H_MAX) &
                                    (s_vals >= EXCL_GREEN_SAT))
                        if green_px.sum() / max(len(h_vals), 1) > EXCL_GREEN_FRAC:
                            continue
                        # Y 轴质心过滤：排除图像最底部的地面/背景
                        ys_m = np.where(m)[0]
                        if ys_m.mean() > GRAPE_MAX_Y_FRAC * H_fr:
                            continue
                        grape_masks.append(m.astype(bool))

                    # 第一轮 tight 合并 + 第二轮质心聚类
                    if grape_masks:
                        grape_np = np.stack(grape_masks, axis=0)
                        merged = merge_masks(grape_np, dilate_kernel=DILATE_KERNEL,
                                             min_area=0, max_area=None,
                                             min_berry_ar=0)
                        merged = union_find_merge(merged, eps=BUNCH_EPS)
                        merged = [
                            m for m in merged
                            if int(total_px * MIN_MERGED_FRAC)
                            <= int(m.sum())
                            <= int(total_px * MAX_MERGED_FRAC)
                        ]
                        merged.sort(key=lambda m: int(m.sum()), reverse=True)
                    else:
                        merged = []

                    for i, mask in enumerate(merged):
                        color = PALETTE_BGR[i % len(PALETTE_BGR)]
                        fill_mask(overlay, mask, color, ALPHA_GRAPE)
                        draw_halo_contour(overlay, mask, color, thick=2)
                        draw_label_box(overlay, mask, f"C{i+1}", color,
                                       font_scale=0.7)
                    n_masks = len(merged)

            # ── 计算 FPS ──────────────────────────────────────────────────────
            current_time = time.time()
            fps_times.append(current_time)
            if len(fps_times) > fps_window:
                fps_times.pop(0)
            fps = ((len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                   if len(fps_times) > 1 else 0)

            # ── HUD 文字 ──────────────────────────────────────────────────────
            mode_label = "MODE: GRAPE ONLY [t]" if grape_only_mode else "MODE: ALL  [t]"
            mode_color = (0, 200, 255) if grape_only_mode else (200, 200, 200)

            cv2.putText(overlay,
                        f"Masks: {n_masks}   Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(overlay,
                        "q:quit  s:save  f:fullscreen  t:mode",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                        1, cv2.LINE_AA)
            cv2.putText(overlay,
                        mode_label,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2,
                        cv2.LINE_AA)

            # 右上角显示 FPS（青色）
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            fps_x = overlay.shape[1] - text_size[0] - 10
            cv2.putText(overlay,
                        fps_text,
                        (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                grape_only_mode = not grape_only_mode
                label = "仅葡萄" if grape_only_mode else "全局"
                print(f"[→] 切换模式 → {label}")
            elif key == ord('s'):
                save_count += 1
                save_path = os.path.join(OUTPUT_DIR,
                                         f"camera_frame_{save_count:04d}.png")
                cv2.imwrite(save_path, overlay)
                print(f"[✓] 已保存: {save_path}")
            elif key == ord('f'):
                cur = cv2.getWindowProperty(window_name,
                                            cv2.WND_PROP_FULLSCREEN)
                next_state = (cv2.WINDOW_FULLSCREEN
                              if cur != cv2.WINDOW_FULLSCREEN
                              else cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(window_name,
                                      cv2.WND_PROP_FULLSCREEN, next_state)

    finally:
        if use_realsense and pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # 恢复 iio-sensor-proxy
        if iio_stopped:
            try:
                subprocess.run(
                    ["sudo", "systemctl", "start", "iio-sensor-proxy"],
                    capture_output=True, timeout=5
                )
                print("[✓] iio-sensor-proxy 已恢复")
            except Exception:
                pass
        print(f"\n[✓] 已退出  共处理 {frame_count} 帧，保存 {save_count} 张")


def main():
    # 1. 确认输入图像
    if not Path(IMAGE_PATH).exists():
        print(f"[✗] 找不到输入图像: {IMAGE_PATH}")
        sys.exit(1)

    # 2. 下载模型权重
    download_model()

    # 3. 导入 FastSAM（通过 ultralytics）
    try:
        from ultralytics import FastSAM
    except ImportError:
        print("[✗] ultralytics 未安装，请先运行: pip install ultralytics")
        sys.exit(1)

    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 尝试配置中文字体
    plt.rcParams['font.sans-serif'] = [
        'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    import torch as _torch

    # 4. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 加载模型 & 推理
    print(f"[→] 加载模型 {MODEL_FILE} ...")
    model = FastSAM(MODEL_FILE)

    print(f"[→] 推理 conf={CONF}, iou={IOU}, imgsz={IMGSZ} ...")
    results = model(
        IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
    )

    # 6. 获取所有 mask
    result = results[0]
    ann    = result.masks

    if ann is None or ann.data is None or len(ann.data) == 0:
        print("[!] 未检测到任何实例，请尝试降低 CONF 阈值")
        return

    masks_np = ann.data.bool().cpu().numpy()   # bool (N, H, W)
    n_raw    = len(masks_np)
    print(f"[→] 原始检测到 {n_raw} 个 mask")

    # 立即释放 GPU / tensor 内存（250 个 mask 的 plot() 会 OOM，直接跳过）
    del ann, result, results
    _torch.cuda.empty_cache()
    gc.collect()

    # 8. 读图 & 计算过滤参数
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W      = image_bgr.shape[:2]
    total_area = H * W
    max_area   = int(total_area * MAX_AREA_FRAC)

    # 9. 膨胀法聚类（带背景大块过滤）
    print(f"[→] 聚类合并（第1轮tight-dilate={DILATE_KERNEL}px + 第2轮centroid-merge EPS={BUNCH_EPS}px） ...")

    # 9. 预分离 mask（在聚类前区分葡萄粒型 / 茎型 / 丢弃）
    berry_max = int(total_area * MAX_AREA_FRAC)
    stem_max  = int(total_area * STEM_MAX_FRAC)

    # 计算图像 HSV（用于颜色过滤）
    image_hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel   = image_hsv[..., 0]   # Hue  0-179
    s_channel   = image_hsv[..., 1]   # Saturation 0-255
    v_channel   = image_hsv[..., 2]   # Value 0-255

    berry_masks_list: list = []   # 葡萄粒型（面积适中、颜色匹配、形状接近圆）
    stem_masks_list:  list = []   # 茎型（细长小 mask，不需满足颜色过滤）

    for m in masks_np:
        area = int(m.sum())
        if area < MIN_AREA:
            continue

        ar = aspect_ratio(m)     # 短/长边比

        # ── 茎型：极细长 + 棕褐色双重验证 ────────────────────────────────────
        if ar < STEM_ASPECT and area <= stem_max:
            _h = h_channel[m]
            _s = s_channel[m]
            # 棕褐色：H in [5,25] (10-50° 标准=棕/橙棕), S>30
            brown_px = ((_h >= 5) & (_h <= 25) & (_s >= 30))
            if brown_px.sum() / max(len(_h), 1) >= STEM_COLOR_FRAC:
                stem_masks_list.append(m.astype(bool))
            continue

        # ── 面积上限（剔除叶片大块）───────────────────────────────────────────
        if area > berry_max:
            continue

        # ── 形状过滤（剔除细长非球形 mask）──────────────────────────────────
        if ar < MIN_BERRY_AR:
            continue

        # ── 颜色过滤：排除绿叶 mask（兼容深蓝/紫/红各色葡萄）───────────────────
        h_vals = h_channel[m]
        s_vals = s_channel[m]
        # 绿叶：H in [EXCL_GREEN_H_MIN, EXCL_GREEN_H_MAX] + 高饱和度
        green_px = ((h_vals >= EXCL_GREEN_H_MIN) & (h_vals <= EXCL_GREEN_H_MAX)
                    & (s_vals >= EXCL_GREEN_SAT))
        if green_px.sum() / max(len(h_vals), 1) > EXCL_GREEN_FRAC:
            continue   # 主体为绿叶→丢弃

        # ── Y 轴质心过滤：排除底部背景（人物/地面）mask ──────────────────────
        ys_m = np.where(m)[0]
        if ys_m.mean() > GRAPE_MAX_Y_FRAC * H:
            continue

        berry_masks_list.append(m.astype(bool))

    print(f"[→] 预分离：葡萄粒型 {len(berry_masks_list)} 个 / 茎型 {len(stem_masks_list)} 个")

    # 10. 葡萄粒聚类（两轮）
    if berry_masks_list:
        berry_np = np.stack(berry_masks_list, axis=0)
        merged_berries = merge_masks(berry_np, dilate_kernel=DILATE_KERNEL,
                                     min_area=0, max_area=None, min_berry_ar=0)
        print(f"[→] 第一轮: {len(merged_berries)} 个小簇")
        merged_berries = union_find_merge(merged_berries, eps=BUNCH_EPS)
        # 第三轮：对合并后过宽的簇沿 X 轴谷值分裂（解决两串极近时被合并的问题）
        merged_berries = split_wide_clusters(merged_berries, W, split_width_frac=0.35)
        grape_clusters = [
            m for m in merged_berries
            if int(total_area * MIN_MERGED_FRAC) <= int(m.sum())
            <= int(total_area * MAX_MERGED_FRAC)
        ]
        grape_clusters.sort(key=lambda m: int(m.sum()), reverse=True)
    else:
        grape_clusters = []
    print(f"[→] 葡萄串簇: {len(grape_clusters)}")

    # 11. 茎部聚类（独立路径，不经过 berry 过滤）
    if stem_masks_list:
        stem_np = np.stack(stem_masks_list, axis=0)
        stem_clusters = merge_masks(stem_np, dilate_kernel=DILATE_KERNEL,
                                    min_area=0, max_area=None, min_berry_ar=0)
        # 茎可能由多个断裂小 mask 组成，用宽 EPS 再合一次
        stem_clusters = union_find_merge(stem_clusters, eps=BUNCH_EPS)
    else:
        stem_clusters = []
    print(f"[→] 茎/细条: {len(stem_clusters)}")

    # 12. 绘制可视化结果（全程 BGR）
    overlay_bgr = image_bgr.copy()   # 彩色填充图
    contour_bgr = image_bgr.copy()   # 轮廓 + 标签图

    for i, mask in enumerate(grape_clusters):
        color = PALETTE_BGR[i % len(PALETTE_BGR)]
        fill_mask(overlay_bgr, mask, color, ALPHA_GRAPE)
        draw_halo_contour(overlay_bgr, mask, color, thick=2)
        draw_halo_contour(contour_bgr, mask, color, thick=3)
        draw_label_box(contour_bgr, mask, f"C{i+1}", color)

    for j, mask in enumerate(stem_clusters):
        fill_mask(overlay_bgr, mask, STEM_BGR, ALPHA_STEM)
        draw_halo_contour(overlay_bgr, mask, STEM_BGR, thick=2)
        draw_halo_contour(contour_bgr, mask, STEM_BGR, thick=2)
        draw_label_box(contour_bgr, mask, f"S{j+1}", STEM_BGR, font_scale=0.8)

    # 13. 保存单文件
    cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay_v2.png"),  overlay_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "contour_v2.png"),  contour_bgr)

    # 14. 三图对比（matplotlib）
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    contour_rgb = cv2.cvtColor(contour_bgr, cv2.COLOR_BGR2RGB)

    stem_note = f" + {len(stem_clusters)} 茎" if stem_clusters else ""
    titles = [
        "原图",
        f"彩色填充（{len(grape_clusters)} 串{stem_note}）",
        "轮廓 + 标签",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, img, title in zip(axes, [image_rgb, overlay_rgb, contour_rgb], titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    out_compare = os.path.join(OUTPUT_DIR, "comparison_v2.png")
    plt.savefig(out_compare, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 对比图已保存: {out_compare}")

    # 15. 统计
    print("\n── 各实例面积统计 ──")
    for i, m in enumerate(grape_clusters):
        pct = int(m.sum()) / total_area * 100
        print(f"  C{i+1:2d}: {int(m.sum()):8d} px  ({pct:.1f}%)")
    for j, m in enumerate(stem_clusters):
        pct = int(m.sum()) / total_area * 100
        print(f"  S{j+1:2d}: {int(m.sum()):8d} px  ({pct:.1f}%)")
    print(f"──────")
    print(f"  原始 mask={n_raw}  葡萄串={len(grape_clusters)}  茎={len(stem_clusters)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSAM 分割工具")
    parser.add_argument(
        "--mode",
        choices=["image", "camera"],
        default=None,
        help="运行模式：image（图像文件分割）或 camera（D405 实时分割）",
    )
    args = parser.parse_args()

    if args.mode is None:
        print("══════════════════════════════════════")
        print("  FastSAM 分割工具 — 请选择运行模式")
        print("══════════════════════════════════════")
        print("  1. 图像分割   (image)  — 处理静态图片")
        print("  2. 实时摄像头 (camera) — Intel RealSense D405 实时分割")
        print("──────────────────────────────────────")
        choice = input("请输入 1 或 2（默认 1）: ").strip()
        args.mode = "camera" if choice == "2" else "image"
        print()

    if args.mode == "camera":
        run_camera()
    else:
        main()
