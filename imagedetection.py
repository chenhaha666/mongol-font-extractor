# qc_suite.py
# -*- coding: utf-8 -*-
"""
字体位图质量评测（只对齐黑色字形，而不是整张图）
对齐流程：
  1) 墨迹二值化（0=墨迹，255=背景，自动纠正反色，并保留最大连通域）
  2) 矩(面积/主轴/质心)初始对齐：一次性得到缩放 s、旋转 θ、平移 (dx,dy)
  3) 在墨迹边缘上做相位相关细化平移，再 ±2px 网格搜索取 IoU 最高
  4) （可选）局部小角度旋转搜索（±rot_search°）
  5) （可选）自动基线旋转：none/cw90/ccw90 三选一
  6) 计算 IoU、Chamfer、Hausdorff、质心偏移、相似度百分比等
默认行为：
  - ink-only（只看黑色字形）开启；如需灰度 SSIM 比较，加 --no-ink-only
  - 不做镜像翻转；如怀疑图片被镜像，可加 --flip-try（且需 IoU 明显提升才采用）
  - 使用无 GUI 后端，避免 Qt 插件报错
依赖:
  pip install pillow numpy opencv-python matplotlib scikit-image
"""

import os, re, math, argparse, csv, json
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

from glob import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

# ---------- 可选：SSIM ----------
try:
    from skimage.metrics import structural_similarity as ssim_metric
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

HEX_RE = re.compile(r"U\+([0-9A-Fa-f]{4,6})\.png$")

# ========================= 基础工具 =========================
def list_codepoints(img_dir):
    items = []
    for p in sorted(glob(os.path.join(img_dir, "U+*.png"))):
        m = HEX_RE.search(os.path.basename(p))
        if m:
            items.append((int(m.group(1), 16), p))
    return items

def bin_otsu(gray):
    blur = cv2.GaussianBlur(gray, (3,3), 0.5)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def ink_mask(gray, keep_largest=True, min_area=16):
    g = cv2.GaussianBlur(gray, (3,3), 0.5)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = 255 - bw
    if keep_largest:
        inv = 255 - bw
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        if num > 1:
            biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            bw = np.where(labels == biggest, 0, 255).astype(np.uint8)
    if min_area > 0:
        inv = 255 - bw
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                inv[labels == i] = 0
        bw = 255 - inv
    if (bw == 255).sum() < (bw == 0).sum():
        bw = 255 - bw
    return bw

def mask_edges(bw):
    k = np.ones((3,3), np.uint8)
    edge = cv2.morphologyEx(255 - bw, cv2.MORPH_GRADIENT, k)
    return (edge>0).astype(np.uint8)*255

def smart_edges(img):
    vals = np.unique(img)
    if len(vals) <= 3 and 0 in vals and 255 in vals:
        return mask_edges(img)
    v = np.median(img)
    lo = int(max(0, 0.66*v)); hi = int(min(255, 1.33*v))
    return cv2.Canny(img, lo, hi)

def iou_masks(a_bw, b_bw):
    a = (a_bw == 0).astype(np.uint8)
    b = (b_bw == 0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    return (inter/union) if union else 1.0

def soft_iou_masks(a_bw, b_bw, dilate_px=0):
    """soft IoU：两边各自先膨胀 r 像素，再计算 IoU（r=0 等价于普通 IoU）"""
    if dilate_px <= 0:
        return iou_masks(a_bw, b_bw)
    a = (a_bw == 0).astype(np.uint8)
    b = (b_bw == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    a_d = cv2.dilate(a, k); b_d = cv2.dilate(b, k)
    inter = (a_d & b_d).sum()
    union = (a_d | b_d).sum()
    return (inter/union) if union else 1.0

def chamfer_ratio(eA, eB):
    H, W = eA.shape; diag = math.hypot(H, W)
    eA = (eA > 0).astype(np.uint8); eB = (eB > 0).astype(np.uint8)
    dtA = cv2.distanceTransform(255-(eA*255), cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255-(eB*255), cv2.DIST_L2, 3)
    ysA, xsA = np.where(eA>0); ysB, xsB = np.where(eB>0)
    if ysA.size==0 and ysB.size==0: return 0.0
    if ysA.size==0 or ysB.size==0:  return 1.0
    dAB = dtB[ysA, xsA].mean(); dBA = dtA[ysB, xsB].mean()
    return float(0.5*(dAB + dBA) / diag)

def hausdorff_ratio(edgeA, edgeB):
    H, W = edgeA.shape; diag = math.hypot(H, W)
    eA = (edgeA > 0).astype(np.uint8); eB = (edgeB > 0).astype(np.uint8)
    dtA = cv2.distanceTransform(255 - eA*255, cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255 - eB*255, cv2.DIST_L2, 3)
    ysA, xsA = np.where(eA); ysB, xsB = np.where(eB)
    if ysA.size == 0 and ysB.size == 0: return 0.0
    if ysA.size == 0 or ysB.size == 0:  return 1.0
    dAB = dtB[ysA, xsA].max(); dBA = dtA[ysB, xsB].max()
    return float(max(dAB, dBA) / diag)

def hausdorff_ratio_q(edgeA, edgeB, q=1.0):
    """分位 Hausdorff：用 q 分位替代最坏点（q=1 等价于传统 Hausdorff）"""
    H, W = edgeA.shape; diag = math.hypot(H, W)
    eA = (edgeA > 0).astype(np.uint8); eB = (edgeB > 0).astype(np.uint8)
    dtA = cv2.distanceTransform(255 - eA*255, cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255 - eB*255, cv2.DIST_L2, 3)
    ysA, xsA = np.where(eA); ysB, xsB = np.where(eB)
    if ysA.size == 0 and ysB.size == 0: return 0.0
    if ysA.size == 0 or ysB.size == 0:  return 1.0
    dAB = np.quantile(dtB[ysA, xsA], q)
    dBA = np.quantile(dtA[ysB, xsB], q)
    return float(max(dAB, dBA) / diag)

def center_of_mass_from_bw(bw):
    ink = (255 - bw)//255
    m = cv2.moments(ink.astype(np.uint8))
    if m["m00"] == 0: return None
    return (m["m10"]/m["m00"], m["m01"]/m["m00"])

def bbox_from_bw(bw):
    m = (bw == 0); ys, xs = np.where(m)
    if ys.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def eff_stroke_width_from_bw(bw):
    cnts, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    area = (bw == 0).sum()
    if per <= 1e-6: return 0.0
    return float(2.0 * area / per)

# ========================= 字体渲染 =========================
def _load_font(font_path, size):
    path = font_path; index = 0
    if "#" in font_path:
        path, idx = font_path.split("#", 1)
        try: index = int(idx)
        except: index = 0
    return ImageFont.truetype(path, size=size, index=index)

def render_baseline(ch, font_path, out_size=128, render_scale=8, margin_ratio=0.12, rotate="none"):
    R = out_size * max(2, int(render_scale))
    img = Image.new("L", (R,R), 255); draw = ImageDraw.Draw(img)
    font = _load_font(font_path, size=int(R*0.86))
    bbox_ = draw.textbbox((0,0), ch, font=font)
    if not bbox_ or (bbox_[2]-bbox_[0]==0) or (bbox_[3]-bbox_[1]==0):
        return Image.fromarray(np.full((out_size,out_size),255,np.uint8))
    w,h = bbox_[2]-bbox_[0], bbox_[3]-bbox_[1]
    x = (R-w)//2 - bbox_[0]; y = (R-h)//2 - bbox_[1]
    draw.text((x,y), ch, 0, font=font)
    arr = np.array(img)
    if rotate=="cw90": arr=cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate=="ccw90": arr=cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ys,xs = np.where(arr<250)
    if ys.size==0: return Image.fromarray(np.full((out_size,out_size),255,np.uint8))
    y0,x0,y1,x1 = ys.min(), xs.min(), ys.max(), xs.max()
    cropped = arr[y0:y1+1, x0:x1+1]
    margin = int(round(out_size*margin_ratio)); pad = max(1, out_size-2*margin)
    ch_,cw_ = cropped.shape
    if ch_>=cw_: new_h=pad; new_w=max(1,int(cw_*pad/ch_))
    else:       new_w=pad; new_h=max(1,int(ch_*pad/cw_))
    resized = cv2.resize(cropped, (new_w,new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_size,out_size),255,np.uint8)
    y_off = margin + (pad-new_h)//2; x_off = margin + (pad-new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return Image.fromarray(canvas)

# ========================= 对齐（严格按墨迹） =========================
def _place_center(canvas_shape, small):
    H,W = canvas_shape; h,w = small.shape
    canvas = np.full((H,W), 255, np.uint8)
    y = max(0, (H - h)//2); x = max(0, (W - w)//2)
    canvas[y:y+min(h,H), x:x+min(w,W)] = small[:min(h,H), :min(w,W)]
    return canvas

def rotate_keep_size(img, angle_deg, border=255):
    H,W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W,H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)

def _scale_to_bbox_by_bw(base_bw, test_gray, test_bw):
    bbA = bbox_from_bw(base_bw); bbB = bbox_from_bw(test_bw)
    if not bbA or not bbB:
        return test_gray, test_bw, 1.0
    hA, wA = bbA[3]-bbA[1]+1, bbA[2]-bbA[0]+1
    hB, wB = bbB[3]-bbB[1]+1, bbB[2]-bbB[0]+1
    s = max(hA, wA) / max(hB, wB) if max(hB,wB)>0 else 1.0
    s = 1.0 if s<=0 else s
    if 0.33 < s < 3.0:
        g2 = cv2.resize(test_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(test_bw,   None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        g2 = _place_center(base_bw.shape, g2)
        b2 = _place_center(base_bw.shape, b2)
        return g2, b2, float(s)
    return test_gray, test_bw, 1.0

def _phase_shift_on_bw_edges(base_bw, test_gray, test_bw, max_shift=512):
    eA = (mask_edges(base_bw)>0).astype(np.float32); eA -= eA.mean()
    eB = (mask_edges(test_bw)>0).astype(np.float32); eB -= eB.mean()
    (shift_y, shift_x), _ = cv2.phaseCorrelate(eB, eA)
    dx = int(np.clip(shift_x, -max_shift, max_shift))
    dy = int(np.clip(shift_y, -max_shift, max_shift))
    M = np.float32([[1,0,dx],[0,1,dy]])
    g2 = cv2.warpAffine(test_gray, M, (test_gray.shape[1], test_gray.shape[0]),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    b2 = cv2.warpAffine(test_bw,   M, (test_bw.shape[1],   test_bw.shape[0]),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return g2, b2, dx, dy

def _refine_local_bw(base_bw, img_gray, img_bw, dx0, dy0, win=2):
    dx0 = int(round(dx0)); dy0 = int(round(dy0)); win = int(round(win))
    best_iou = iou_masks(base_bw, img_bw)
    best = (img_gray, img_bw, dx0, dy0, best_iou)
    for ddx in range(dx0 - win, dx0 + win + 1):
        for ddy in range(dy0 - win, dy0 + win + 1):
            M = np.float32([[1, 0, ddx], [0, 1, ddy]])
            g2 = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            b2 = cv2.warpAffine(img_bw,   M, (img_bw.shape[1],   img_bw.shape[0]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            iou = iou_masks(base_bw, b2)
            if iou > best_iou:
                best = (g2, b2, ddx, ddy, iou)
                best_iou = iou
    return best  # (gray, bw, dx, dy, iou)

def _orientation_angle_from_bw(bw):
    ink = (255 - bw) // 255
    m = cv2.moments(ink.astype(np.uint8))
    mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
    if abs(mu20 - mu02) < 1e-9 and abs(mu11) < 1e-9:
        return 0.0
    return 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))

def _warp_by_M(gray, bw, M, border=255):
    H, W = gray.shape
    g2 = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    b2 = cv2.warpAffine(bw,   M, (W, H), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return g2, b2

def align_by_moments(base_gray, test_gray):
    base_bw = ink_mask(base_gray)
    test_bw = ink_mask(test_gray)

    areaA = float((base_bw == 0).sum())
    areaB = float((test_bw == 0).sum())
    s = 1.0 if areaB <= 1 else math.sqrt(max(areaA,1.0)/areaB)

    angA = _orientation_angle_from_bw(base_bw)
    angB = _orientation_angle_from_bw(test_bw)
    ang = (angA - angB)
    ang_deg = math.degrees(ang)

    cA = center_of_mass_from_bw(base_bw)
    cB = center_of_mass_from_bw(test_bw)
    if (cA is None) or (cB is None):
        return test_gray, test_bw, 1.0, 0.0, 0, 0

    M = cv2.getRotationMatrix2D((cB[0], cB[1]), ang_deg, s)
    M[0, 2] += (cA[0] - cB[0])
    M[1, 2] += (cA[1] - cB[1])
    g2, b2 = _warp_by_M(test_gray, test_bw, M)
    dx = float(M[0,2]); dy = float(M[1,2])
    return g2, b2, s, ang_deg, dx, dy

def _local_rot_search(base_bw, img_gray, img_bw, rot_search=0.0, step=0.5, max_shift=512):
    rot_search = float(max(0.0, rot_search))
    if rot_search <= 1e-6:
        return img_gray, img_bw, 0.0, 0, 0, iou_masks(base_bw, img_bw)

    best = (img_gray, img_bw, 0.0, 0, 0, iou_masks(base_bw, img_bw))
    steps = int(max(1, round(rot_search/step)))
    angles = [k*step for k in range(-steps, steps+1)]
    for da in angles:
        if abs(da) < 1e-9:
            continue
        g_rot = rotate_keep_size(img_gray, da, border=255)
        b_rot = rotate_keep_size(img_bw,   da, border=255)
        g2, b2, dx1, dy1 = _phase_shift_on_bw_edges(base_bw, g_rot, b_rot, max_shift=max_shift)
        g3, b3, dx2, dy2, iou = _refine_local_bw(base_bw, g2, b2, dx1, dy1, win=2)
        if iou > best[5]:
            best = (g3, b3, da, dx2, dy2, iou)
    return best  # (gray, bw, delta_angle, dx, dy, iou)

def auto_align_ink(base_gray, test_gray, try_flip=False, max_shift=512,
                   min_flip_gain=0.12, rot_search=0.0):
    base_bw = ink_mask(base_gray)
    flips = [("none", test_gray)]
    if try_flip:
        flips += [("flipX", cv2.flip(test_gray,1)),
                  ("flipY", cv2.flip(test_gray,0)),
                  ("flipXY", cv2.flip(test_gray,-1))]

    best = None
    none_cand = None

    for tag, g0 in flips:
        g1, b1, s0, ang0, dx0, dy0 = align_by_moments(base_gray, g0)
        g1, b1, s_box = _scale_to_bbox_by_bw(base_bw, g1, b1)
        s0 *= s_box
        g2, b2, dx, dy = _phase_shift_on_bw_edges(base_bw, g1, b1, max_shift=max_shift)
        dx += dx0; dy += dy0
        g3, b3, dx2, dy2, _ = _refine_local_bw(base_bw, g2, b2, dx, dy, win=2)
        g4, b4, da, ddx, ddy, iou = _local_rot_search(base_bw, g3, b3, rot_search=rot_search, step=0.5, max_shift=max_shift)
        cand = {"gray": g4, "bw": b4,
                "dx": ddx, "dy": ddy,
                "scale": s0, "flip": tag,
                "angle": ang0 + da, "iou": iou}
        if tag == "none":
            none_cand = cand
        if (best is None) or (iou > best["iou"]):
            best = cand

    if best["flip"] != "none" and none_cand is not None and best["iou"] < none_cand["iou"] + min_flip_gain:
        best = none_cand
    return best["gray"], best["bw"], best["dx"], best["dy"], best["scale"], best["flip"], best["angle"], best["iou"]

# ========================= 面板可视化（6格） =========================
def make_panel(imgA, imgB, metrics_text):
    H,W = imgA.shape
    diff = cv2.absdiff(imgA, imgB)
    diff_norm = (diff * (255.0/max(1,diff.max()))).astype(np.uint8)
    diff_heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    eA = smart_edges(imgA); eB = smart_edges(imgB)
    overlay = np.full((H,W,3), 255, np.uint8)
    overlay[eA>0] = (0,255,0); overlay[eB>0] = (255,0,0)
    overlay[(eA>0)&(eB>0)] = (255,255,0)

    dtA = cv2.distanceTransform(255-(eA>0).astype(np.uint8)*255, cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform(255-(eB>0).astype(np.uint8)*255, cv2.DIST_L2, 3)
    dtd = np.abs(dtA-dtB); dtd = (dtd / (dtd.max()+1e-6) * 255).astype(np.uint8)
    dtd_heat = cv2.applyColorMap(dtd, cv2.COLORMAP_JET)

    contours_img = np.full((H,W,3), 255, np.uint8)
    cntsA,_ = cv2.findContours(255 - bin_otsu(imgA), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntsB,_ = cv2.findContours(255 - bin_otsu(imgB), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contours_img, cntsA, -1, (0,128,255), 1)
    cv2.drawContours(contours_img, cntsB, -1, (255,64,64), 1)

    row1 = np.hstack([cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR), diff_heat])
    row2 = np.hstack([overlay, dtd_heat, contours_img])
    panel = np.vstack([row1, row2]).copy()
    y0 = 20
    for t in metrics_text.split("\n"):
        cv2.putText(panel, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        y0 += 18
    return panel

# ========================= 曲率与笔宽（基于墨迹） =========================
def contour_longest_from_bw(bw):
    cnts, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def curvature_profile_from_contour(cnt, samples=256):
    pts = cnt[:, 0, :].astype(np.float32)
    seg = np.sqrt(((pts[1:] - pts[:-1])**2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        t = np.linspace(0,1,len(pts)); return t, np.zeros_like(t)
    t = np.linspace(0, s[-1], samples)
    xs = np.interp(t, s, pts[:,0]); ys = np.interp(t, s, pts[:,1])
    dx = np.gradient(xs); dy = np.gradient(ys)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    k = np.abs(dx*ddy - dy*ddx) / (dx*dx + dy*dy + 1e-9)**1.5
    if k.max() > 0: k = k / k.max()
    return np.linspace(0,1,samples), k

def curvature_compare_plot_from_bw(bwA, bwB, out_png):
    cA = contour_longest_from_bw(bwA); cB = contour_longest_from_bw(bwB)
    if cA is None or cB is None:
        return None, None
    sA, kA = curvature_profile_from_contour(cA, 256)
    sB, kB = curvature_profile_from_contour(cB, 256)
    plt.figure(); plt.plot(sA, kA, label="baseline")
    plt.plot(sB, kB, label="output", linestyle="--")
    plt.xlabel("normalized arclength"); plt.ylabel("normalized curvature")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()
    l2 = float(np.sqrt(np.mean((kA - kB)**2)))
    peakA = float(sA[int(np.argmax(kA))]); peakB = float(sB[int(np.argmax(kB))])
    peak_shift = float(abs(peakA - peakB))
    return {"curv_l2": l2, "peak_shift": peak_shift}, out_png

def stroke_width_stats_from_bw(bw):
    dt = cv2.distanceTransform(255 - bw, cv2.DIST_L2, 3)
    widths = (dt[dt > 0.5] * 2.0)
    if widths.size == 0:
        return None, None
    return widths, {"sw_mean": float(np.mean(widths)),
                    "sw_median": float(np.median(widths)),
                    "sw_std": float(np.std(widths))}

# ========================= 相似度百分比（加可调权重/尺度） =========================
def similarity_percent_weighted(ssim_val, iou_val, chamfer, hausdorff, centroid_px,
                                w_ssim=0.35, w_iou=0.35, w_chamfer=0.10, w_haus=0.10, w_centroid=0.10,
                                ref_chamfer=0.03, ref_haus=0.03, ref_centroid=3.0):
    comps, weights = [], []
    if ssim_val is not None and np.isfinite(ssim_val):
        comps.append(float(np.clip(ssim_val, 0.0, 1.0))); weights.append(float(w_ssim))
    if iou_val is not None and np.isfinite(iou_val):
        comps.append(float(np.clip(iou_val, 0.0, 1.0))); weights.append(float(w_iou))
    if chamfer is not None and np.isfinite(chamfer):
        comps.append(1.0 - float(np.clip(chamfer/max(1e-9, ref_chamfer), 0.0, 1.0))); weights.append(float(w_chamfer))
    if hausdorff is not None and np.isfinite(hausdorff):
        comps.append(1.0 - float(np.clip(hausdorff/max(1e-9, ref_haus), 0.0, 1.0))); weights.append(float(w_haus))
    if centroid_px is not None and np.isfinite(centroid_px):
        comps.append(1.0 - float(np.clip(centroid_px/max(1e-9, ref_centroid), 0.0, 1.0))); weights.append(float(w_centroid))
    if not comps:
        return 0.0
    weights = np.array(weights, dtype=float)
    if weights.sum() <= 1e-9:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    score = float(np.clip(np.dot(weights, np.array(comps, dtype=float)), 0.0, 1.0))
    return round(score * 100.0, 2)

# ========================= 图表工具 =========================
def save_hist(data, title, fname, outdir, bins=30):
    data = np.array([x for x in data if x is not None and np.isfinite(x)])
    if data.size==0: return
    plt.figure(); plt.hist(data, bins=bins); plt.title(title)
    plt.xlabel(title); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fname)); plt.close()

def save_scatter(x, y, xlabel, ylabel, title, fname, outdir):
    x = np.array([a for a in x if a is not None and np.isfinite(a)])
    y = np.array([b for b in y if b is not None and np.isfinite(b)])
    if x.size==0 or y.size==0: return
    n = min(len(x), len(y))
    plt.figure(); plt.scatter(x[:n], y[:n], s=10)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fname)); plt.close()

def contact_sheet(images, out_path, cols=6, bg=255):
    imgs = [cv2.imread(p) for p in images if p and os.path.isfile(p)]
    if not imgs: return
    h, w, _ = imgs[0].shape
    rows_ct = int(np.ceil(len(imgs)/cols))
    sheet = np.full((rows_ct*h, cols*w, 3), bg, np.uint8)
    for idx, im in enumerate(imgs):
        r = idx//cols; c = idx%cols
        sheet[r*h:(r+1)*h, c*w:(w*(c+1))] = im
    cv2.imwrite(out_path, sheet)

# ========================= 主过程 =========================
def main():
    ap = argparse.ArgumentParser(description="字体位图质量评测（墨迹对齐）")
    ap.add_argument("--font", required=True, help="TTF/OTF 路径（TTC 可用 #index）")
    ap.add_argument("--images", required=True, help="导出PNG目录（U+XXXX.png）")
    ap.add_argument("--out", default="qc_out", help="输出目录")
    ap.add_argument("--size", type=int, default=128, help="基准渲染尺寸")
    ap.add_argument("--render-scale", type=int, default=8, help="基准渲染超采样倍数")
    ap.add_argument("--margin", type=float, default=0.12, help="边距比例（与导出保持一致）")
    ap.add_argument("--rotate", choices=["none","cw90","ccw90"], default="none")
    ap.add_argument("--topk", type=int, default=30, help="Top-K 最差样例拼贴")

    # 对齐/比较选项
    ap.add_argument("--align", choices=["auto","none"], default="auto", help="是否自动对齐（默认 auto）")
    ap.add_argument("--max-shift", type=int, default=512, help="对齐允许的最大平移像素")
    ap.add_argument("--flip-try", action="store_true", help="尝试镜像（默认不翻转；仅在 IoU 显著提升时采用）")
    ap.add_argument("--min-flip-gain", type=float, default=0.12, help="翻转采用的最小 IoU 提升阈值")
    ap.add_argument("--rot-search", type=float, default=0.0, help="小角度旋转搜索范围（度），如 2 表示 ±2°（通常无需开启）")

    # 自动基线旋转
    ap.add_argument("--auto-rotate-baseline", action="store_true", help="自动选择基线旋转方向（none/cw90/ccw90）")

    # 只看墨迹：默认开启；如需灰度 SSIM 可用 --no-ink-only 关闭
    ap.add_argument("--ink-only", dest="ink_only", action="store_true", help="只按黑色墨迹比较（默认开启）")
    ap.add_argument("--no-ink-only", dest="ink_only", action="store_false", help="关闭墨迹模式，允许灰度SSIM参与")
    ap.set_defaults(ink_only=True)

    # 阈值（用于 pass/fail 判定）
    ap.add_argument("--ssim-min", type=float, default=0.97)
    ap.add_argument("--iou-min", type=float, default=0.93)
    ap.add_argument("--chamfer-max", type=float, default=0.02)
    ap.add_argument("--centroid-max", type=float, default=1.5)

    # —— 度量的“宽容度”与权重（新增） ——
    ap.add_argument("--iou-soft", type=int, default=0, help="soft IoU 的膨胀半径(px)，0=关闭")
    ap.add_argument("--hausdorff-q", type=float, default=1.0, help="Hausdorff 分位（0.90~1.00，1.0=最大值）")

    ap.add_argument("--w-ssim", type=float, default=0.35, help="综合分中 SSIM 的权重")
    ap.add_argument("--w-iou", type=float, default=0.35, help="综合分中 IoU 的权重")
    ap.add_argument("--w-chamfer", type=float, default=0.10, help="综合分中 Chamfer 的权重")
    ap.add_argument("--w-haus", type=float, default=0.10, help="综合分中 Hausdorff 的权重")
    ap.add_argument("--w-centroid", type=float, default=0.10, help="综合分中 质心 的权重")

    ap.add_argument("--ref-chamfer", type=float, default=0.03, help="Chamfer 归一化参考尺度")
    ap.add_argument("--ref-haus", type=float, default=0.03, help="Hausdorff 归一化参考尺度")
    ap.add_argument("--ref-centroid", type=float, default=3.0, help="质心像素偏移归一化参考尺度")

    args = ap.parse_args()

    font_probe = args.font.split("#")[0]
    if not os.path.isfile(font_probe):
        raise FileNotFoundError(f"字体文件不存在：{os.path.abspath(font_probe)}")

    os.makedirs(args.out, exist_ok=True)
    vis_dir = os.path.join(args.out, "per_glyph"); os.makedirs(vis_dir, exist_ok=True)

    pairs = list_codepoints(args.images)
    if not pairs:
        print("未找到 U+XXXX.png 文件"); return

    rows = []; panel_paths = []

    for code, png_path in pairs:
        ch = chr(code)
        try:
            test = Image.open(png_path).convert("L")
        except Exception as e:
            print(f"{os.path.basename(png_path)} 打不开：{e}"); continue
        raw_test_g = np.array(test)

        # 基线渲染 + 自动基线旋转（可选）
        def run_once(base_gray, test_gray):
            if base_gray.shape != test_gray.shape:
                test_gray = cv2.resize(test_gray, (base_gray.shape[1], base_gray.shape[0]), interpolation=cv2.INTER_AREA)
            # 预对齐前的 soft IoU（直观对比“对齐是否有效”）
            iou_pre = soft_iou_masks(ink_mask(base_gray), ink_mask(test_gray), args.iou_soft)
            if args.align == "auto":
                test_aligned_g, test_bw_aligned, dx, dy, scl, flip_tag, angle, _ = auto_align_ink(
                    base_gray, test_gray, try_flip=args.flip_try,
                    max_shift=args.max_shift, min_flip_gain=args.min_flip_gain,
                    rot_search=args.rot_search
                )
            else:
                test_aligned_g = test_gray
                test_bw_aligned = ink_mask(test_aligned_g)
                dx = dy = 0; scl = 1.0; flip_tag = "none"; angle = 0.0
            return dict(test_g=test_aligned_g, test_bw=test_bw_aligned,
                        iou_pre=iou_pre, dx=dx, dy=dy, scl=scl, flip=flip_tag, angle=angle)

        rot_list = ["none","cw90","ccw90"] if args.auto_rotate_baseline else [args.rotate]
        best_pack = None
        best_base_g = None

        for rot_opt in rot_list:
            base = render_baseline(ch, args.font, out_size=args.size,
                                   render_scale=args.render_scale,
                                   margin_ratio=args.margin, rotate=rot_opt)
            base_g = np.array(base)
            pack = run_once(base_g, raw_test_g)
            iou_aligned = soft_iou_masks(ink_mask(base_g), pack["test_bw"], args.iou_soft)
            if (best_pack is None) or (iou_aligned > soft_iou_masks(ink_mask(best_base_g), best_pack["test_bw"], args.iou_soft)):
                best_pack = pack
                best_base_g = base_g

        base_g = best_base_g
        test_g = best_pack["test_g"]
        test_bw_aligned = best_pack["test_bw"]
        dx = best_pack["dx"]; dy = best_pack["dy"]
        scl = best_pack["scl"]; flip_tag = best_pack["flip"]; angle = best_pack["angle"]
        iou_pre = best_pack["iou_pre"]

        # ===== 墨迹（二值） =====
        bwA = ink_mask(base_g)
        bwB = test_bw_aligned  # 已对齐

        # ===== 指标 =====
        ssim_val = None if (args.ink_only or not HAS_SKIMAGE) else float(ssim_metric(base_g, test_g, data_range=255))
        iou_val = float(soft_iou_masks(bwA, bwB, args.iou_soft))
        eA = mask_edges(bwA) if args.ink_only else smart_edges(base_g)
        eB = mask_edges(bwB) if args.ink_only else smart_edges(test_g)
        chamfer = float(chamfer_ratio(eA, eB))
        if args.hausdorff_q >= 0.9999:
            haus = float(hausdorff_ratio(eA, eB))
        else:
            haus = float(hausdorff_ratio_q(eA, eB, args.hausdorff_q))
        cA = center_of_mass_from_bw(bwA); cB = center_of_mass_from_bw(bwB)
        centroid = float(math.hypot(cA[0]-cB[0], cA[1]-cB[1])) if (cA and cB) else None

        bbA = bbox_from_bw(bwA); bbB = bbox_from_bw(bwB)
        if bbA and bbB:
            H, W = bwA.shape
            x0A,y0A,x1A,y1A = bbA; x0B,y0B,x1B,y1B = bbB
            margA = (x0A/W, y0A/H, (W-1-x1A)/W, (H-1-y1A)/H)
            margB = (x0B/W, y0B/H, (W-1-x1B)/W, (H-1-y1B)/H)
            margin_diff = float(np.mean(np.abs(np.array(margA) - np.array(margB))))
        else:
            margin_diff = None

        swA_vals, swA_stats = stroke_width_stats_from_bw(bwA)
        swB_vals, swB_stats = stroke_width_stats_from_bw(bwB)
        if swA_vals is None or swB_vals is None:
            sw_kl = None
        else:
            lo = float(min(swA_vals.min(), swB_vals.min()))
            hi = float(max(swA_vals.max(), swB_vals.max()))
            hA, edges_h = np.histogram(swA_vals, bins=30, range=(lo, hi), density=True)
            hB, _ = np.histogram(swB_vals, bins=edges_h, density=True)
            eps = 1e-9; P = hA + eps; Q = hB + eps
            sw_kl = float(np.sum(P * np.log(P / Q)))

        curv_metrics, curv_png = curvature_compare_plot_from_bw(
            bwA, bwB, os.path.join(vis_dir, f"U+{code:04X}_curvature.png")
        )

        sim_pct = similarity_percent_weighted(
            ssim_val, iou_val, chamfer, haus, centroid,
            args.w_ssim, args.w_iou, args.w_chamfer, args.w_haus, args.w_centroid,
            args.ref_chamfer, args.ref_haus, args.ref_centroid
        )

        # ===== 面板 =====
        metrics_lines = [
            f"U+{code:04X}  {'('+ch+')' if code<=0xFFFF else ''}",
            f"Similarity: {sim_pct:.2f}%",
            f"Aligned shift(dx,dy)=({dx},{dy}), scale={scl:.3f}, angle={angle:.1f}°, flip={flip_tag}, IoU_pre={iou_pre:.4f}",
            f"SSIM: {'n/a(ink-only)' if (args.ink_only or not HAS_SKIMAGE) else f'{ssim_val:.4f}'}",
            f"IoU (soft r={args.iou_soft}): {iou_val:.4f}",
            f"Chamfer(diag-ratio): {chamfer:.4f}",
            f"Hausdorff(q={args.hausdorff_q:.2f}, diag): {haus:.4f}",
            f"Centroid offset(px): {centroid:.3f}" if centroid is not None else "Centroid offset: n/a",
        ]
        panel = make_panel(bwA if args.ink_only else base_g,
                           bwB if args.ink_only else test_g,
                           "\n".join(metrics_lines))
        out_panel = os.path.join(vis_dir, f"U+{code:04X}.png"); cv2.imwrite(out_panel, panel); panel_paths.append(out_panel)

        rows.append({
            "code": code,
            "panel": out_panel,
            "align_dx": dx, "align_dy": dy, "align_scale": round(scl, 6),
            "align_angle": round(angle, 3), "align_flip": flip_tag,
            "ssim": "" if (ssim_val is None) else round(ssim_val, 6),
            "iou": round(iou_val, 6),
            "chamfer_ratio": round(chamfer, 6),
            "hausdorff_ratio": round(haus, 6),
            "centroid_px": "" if centroid is None else round(centroid, 4),
            "margin_diff": "" if margin_diff is None else round(margin_diff, 6),
            "sw_mean_base": "" if (swA_stats is None) else round(swA_stats["sw_mean"], 6),
            "sw_mean_test": "" if (swB_stats is None) else round(swB_stats["sw_mean"], 6),
            "sw_kl": "" if (sw_kl is None) else round(sw_kl, 6),
            "curv_l2": "" if (curv_metrics is None) else round(curv_metrics["curv_l2"], 6),
            "curv_peak_shift": "" if (curv_metrics is None) else round(curv_metrics["peak_shift"], 6),
            "similarity_percent": sim_pct
        })

    # -------- 导出 CSV --------
    csv_path = os.path.join(args.out, "metrics.csv")
    fieldnames = ["code","similarity_percent",
                  "align_dx","align_dy","align_scale","align_angle","align_flip",
                  "ssim","iou","chamfer_ratio","hausdorff_ratio",
                  "centroid_px","margin_diff","sw_mean_base","sw_mean_test",
                  "sw_kl","curv_l2","curv_peak_shift","panel"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames}
            row["code"] = f"U+{r['code']:04X}"; w.writerow(row)

    # -------- 聚合与阈值判定 --------
    def get_vec(key):
        vals = []
        for r in rows:
            v = r.get(key)
            if v=="" or v is None: continue
            if isinstance(v, (int,float)) and not np.isfinite(v): continue
            vals.append(float(v))
        return np.array(vals) if vals else None

    ss = get_vec("ssim"); ii = get_vec("iou"); ch = get_vec("chamfer_ratio")
    hs = get_vec("hausdorff_ratio"); ct = get_vec("centroid_px"); sp = get_vec("similarity_percent")

    pass_count = 0
    for r in rows:
        ok = True
        if HAS_SKIMAGE and (not args.ink_only) and (r["ssim"]!=""):
            ok &= (float(r["ssim"]) >= args.ssim_min)
        ok &= (float(r["iou"]) >= args.iou_min)
        ok &= (float(r["chamfer_ratio"]) <= args.chamfer_max)
        if r["centroid_px"]!="":
            ok &= (float(r["centroid_px"]) <= args.centroid_max)
        r["pass"] = ok
        if ok: pass_count += 1

    summary = {
        "count": len(rows),
        "pass_count": pass_count,
        "pass_rate": round(100.0 * pass_count / max(1,len(rows)), 2),
        "similarity_percent_mean": None if sp is None else round(float(sp.mean()), 3),
        "ssim_mean": None if ss is None else round(float(ss.mean()), 6),
        "iou_mean": None if ii is None else round(float(ii.mean()), 6),
        "chamfer_mean": None if ch is None else round(float(ch.mean()), 6),
        "hausdorff_mean": None if hs is None else round(float(hs.mean()), 6),
        "centroid_mean": None if ct is None else round(float(ct.mean()), 6),
        "ink_only": True if args.ink_only else False,
        "iou_soft": args.iou_soft,
        "hausdorff_q": args.hausdorff_q
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------- 统计图 --------
    save_hist(ii, "IoU", "hist_iou.png", args.out)
    save_hist(ch, "Chamfer ratio", "hist_chamfer.png", args.out)
    save_hist(hs, "Hausdorff ratio", "hist_hausdorff.png", args.out)
    save_hist(ct, "Centroid offset (px)", "hist_centroid.png", args.out)
    save_hist(sp, "Similarity percent", "hist_similarity.png", args.out)

    # -------- Top-K 最差样例拼贴 --------
    K = min(30, len(rows))
    if K > 0:
        worst_iou = sorted(rows, key=lambda r: r["iou"])[:K]
        worst_haus = sorted(rows, key=lambda r: r["hausdorff_ratio"], reverse=True)[:K]
        contact_sheet([r["panel"] for r in worst_iou], os.path.join(args.out, "topk_worst_iou.png"))
        contact_sheet([r["panel"] for r in worst_haus], os.path.join(args.out, "topk_worst_hausdorff.png"))

    # -------- 简易 HTML 报告 --------
    html = []
    html.append("<html><head><meta charset='utf-8'><title>QC Report</title></head><body>")
    html.append("<h1>字体位图质量评测报告（墨迹对齐）</h1>")
    html.append("<h2>概要</h2>")
    html.append(f"<p>总字符数：{summary['count']}，通过：{summary['pass_count']}（通过率 {summary['pass_rate']}%）</p>")
    html.append(f"<p>总体相似度：<b>{summary['similarity_percent_mean']}%</b></p>")
    html.append("<ul>")
    html.append(f"<li>IoU 均值：{summary['iou_mean']}</li>")
    html.append(f"<li>Chamfer 均值：{summary['chamfer_mean']}</li>")
    html.append(f"<li>Hausdorff 均值：{summary['hausdorff_mean']}（q="
                f"{summary['hausdorff_q']}, softIoU r={summary['iou_soft']}）</li>")
    if summary["centroid_mean"] is not None: html.append(f"<li>质心偏移均值：{summary['centroid_mean']} px</li>")
    html.append(f"<li>Ink-only：{summary['ink_only']}</li>")
    html.append("</ul>")
    html.append("<h2>逐字面板</h2><ul>")
    for r in rows:
        codehex = f"U+{r['code']:04X}"
        rel = os.path.relpath(r["panel"], args.out)
        html.append(f"<li>{codehex} — 相似度 {r['similarity_percent']}% — <a href='{rel}' target='_blank'>{rel}</a></li>")
    html.append("</ul>")
    html.append("<p>详表：<a href='metrics.csv' target='_blank'>metrics.csv</a></p>")
    html.append("</body></html>")
    with open(os.path.join(args.out, "report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"\n完成！输出目录：{args.out}")
    print(f"- 总体相似度：{summary['similarity_percent_mean']}%（Ink-only={summary['ink_only']}，softIoU r={args.iou_soft}，Haus q={args.hausdorff_q}）")
    print("查看 report.html / metrics.csv / per_glyph/* 以获取详情。")

if __name__ == "__main__":
    main()
