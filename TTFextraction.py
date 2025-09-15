import os
import io
import csv
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Optional dependencies, only for vector mode =====
try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.boundsPen import BoundsPen
    FONTTOOLS_AVAILABLE = True
    logger.info("fontTools is installed")
except Exception as e:
    FONTTOOLS_AVAILABLE = False
    logger.warning(f"fontTools is not installed: {e}")

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
    logger.info("cairosvg is installed")
except Exception as e:
    CAIROSVG_AVAILABLE = False
    logger.warning(f"cairosvg is not installed: {e}")

# ===== Unicode range: Mongolian U+1820 ~ U+1842 =====
UNICODE_START = 0x1820
UNICODE_END   = 0x1842

# =========================================================
# Utility functions (smooth version)
# =========================================================
def touches_border(img_np, margin_pixels: int = 0) -> bool:
    """Detect if non-white pixels touch the border (works for grayscale/binary)."""
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
    H, W = gray.shape
    mask = gray < 250

    # Optional: Ignore areas a certain number of pixels from the border
    top    = mask[0 + margin_pixels, :].any()    if 0 + margin_pixels < H else False
    bottom = mask[H-1 - margin_pixels, :].any()  if H-1 - margin_pixels >= 0 else False
    left   = mask[:, 0 + margin_pixels].any()    if 0 + margin_pixels < W else False
    right  = mask[:, W-1 - margin_pixels].any()  if W-1 - margin_pixels >= 0 else False
    return top or bottom or left or right


def trim_center_square_smooth(img_np, out_size=128, margin_ratio=0.12):
    """
    Works for grayscale/binary: Crop tightly -> Scale proportionally to (1-2*margin)*out_size -> Center on white background.
    Does not perform final binarization, preserves anti-aliasing information.
    """
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np.copy()

    mask = gray < 250  # Non-white
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = gray[y0:y1+1, x0:x1+1]

    target = out_size
    margin = int(round(target * margin_ratio))
    pad = max(1, target - 2 * margin)

    ch, cw = cropped.shape
    if ch >= cw:
        new_h = pad
        new_w = max(1, int(cw * pad / ch))
    else:
        new_w = pad
        new_h = max(1, int(ch * pad / cw))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target, target), 255, dtype=np.uint8)
    y_off = margin + (pad - new_h) // 2
    x_off = margin + (pad - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def recenter_by_centroid_smart(canvas_np: np.ndarray, keep_gray: bool,
                               threshold_val: int = 200, use_otsu: bool = False) -> np.ndarray:
    """
    Calculate centroid on a "copy" using Otsu, only translate the original image.
    keep_gray=True -> Return grayscale with anti-aliasing; False -> Binarize at the end.
    """
    if canvas_np.ndim == 3:
        gray = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_np.copy()

    # Mask only for positioning
    _tmp = cv2.GaussianBlur(gray, (3, 3), 0.5)
    if use_otsu:
        _, bw_mask = cv2.threshold(_tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw_mask = cv2.threshold(_tmp, threshold_val, 255, cv2.THRESH_BINARY)
    ink = (bw_mask == 0).astype(np.uint8)

    m = cv2.moments(ink)
    if m["m00"] == 0:
        shifted = gray
    else:
        cy = m["m01"] / m["m00"]
        cx = m["m10"] / m["m00"]
        H, W = gray.shape
        dy = int(round((H - 1) / 2 - cy))
        dx = int(round((W - 1) / 2 - cx))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_AREA, borderValue=255)

    if keep_gray:
        return shifted  # Preserve grayscale edges

    # Final step:轻微平滑后做最终二值化
    blur = cv2.GaussianBlur(shifted, (3, 3), 0.5)
    if use_otsu:
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
    return bw


# =========================================================
# Raster mode: PIL grayscale supersampling rendering
# =========================================================
def raster_render_char(ch, font_path, out_size=128, scale=6, threshold=200,
                       rotate_mode='none', use_otsu=True, margin_ratio=0.12, keep_gray=True):
    """
    Grayscale supersampling -> Rotation -> Smooth cropping/scaling/centering -> Centroid alignment -> (Optional) Final binarization
    """
    render_size = out_size * max(2, int(scale))
    # Use large canvas in grayscale mode (255 white, 0 black), PIL rendering with anti-aliasing
    img = Image.new("L", (render_size, render_size), color=255)
    draw = ImageDraw.Draw(img)

    # Automatic font size: try to fill the canvas (with some safety margin)
    safe = 0.86  # Smaller ratio means larger margin
    font_size = int(render_size * safe)
    font = ImageFont.truetype(font_path, size=font_size)

    # Calculate bbox and center the text
    bbox = draw.textbbox((0, 0), ch, font=font)
    if not bbox or (bbox[2]-bbox[0] == 0) or (bbox[3]-bbox[1] == 0):
        return None

    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (render_size - w) // 2 - bbox[0]
    y = (render_size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)  # Grayscale AA rendering

    img_np = np.array(img)

    # Rotation (still in high-resolution grayscale domain)
    if rotate_mode == 'cw90':
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_mode == 'ccw90':
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Cropping/scaling/centering (preserve grayscale edges)
    m = margin_ratio
    for _ in range(3):
        centered = trim_center_square_smooth(img_np, out_size=out_size, margin_ratio=m)
        if centered is None:
            return None
        centered = recenter_by_centroid_smart(centered, keep_gray=keep_gray,
                                              threshold_val=threshold, use_otsu=use_otsu)
        # Increase margin and retry if touching border
        if not touches_border(centered, margin_pixels=0):
            return Image.fromarray(centered)
        m = min(0.3, m + 0.02)

    return Image.fromarray(centered)


# =========================================================
# Vector mode: TTF → SVG → High-resolution PNG → Smooth downsampling (fixed version)
# =========================================================
def glyph_to_svg(ttfont: "TTFont", ch: str, debug_mode=False) -> str | None:
    """Fixed version: Convert single character glyph to SVG (with coordinate flipping)"""
    if "cmap" not in ttfont:
        logger.error(f"Font missing cmap table: {ch}")
        return None
        
    cmap = ttfont["cmap"].getBestCmap()
    char_code = ord(ch)
    gid_name = cmap.get(char_code)
    
    if gid_name is None:
        logger.error(f"No glyph found for character: U+{char_code:04X}")
        return None

    glyph_set = ttfont.getGlyphSet()
    if gid_name not in glyph_set:
        logger.error(f"Glyph not in font: {gid_name}")
        return None

    glyph = glyph_set[gid_name]
    
    # Get font units
    units = ttfont["head"].unitsPerEm if "head" in ttfont else 1000
    
    # Get glyph bounds
    bpen = BoundsPen(glyph_set)
    glyph.draw(bpen)
    
    if not bpen.bounds:
        logger.warning(f"Missing boundary information for U+{char_code:04X}, using default bounds")
        xMin, yMin, xMax, yMax = 0, 0, units, units
    else:
        xMin, yMin, xMax, yMax = bpen.bounds
        logger.debug(f"Bounds for U+{char_code:04X}: xMin={xMin}, yMin={yMin}, xMax={xMax}, yMax={yMax}")

    # Add 10% safety margin
    padding = units * 0.1
    width  = max(1, int(xMax - xMin + 2 * padding))
    height = max(1, int(yMax - yMin + 2 * padding))
    
    # Get path data
    pen = SVGPathPen(glyph_set)
    glyph.draw(pen)
    path_cmd = pen.getCommands()
    
    if not path_cmd:
        logger.error(f"Empty path data for U+{char_code:04X}")
        return None

    # Fix coordinate system transformation: Y-axis flip + offset correction
    transform = f"scale(1, -1) translate({-xMin + padding}, {-yMax - padding})"
    
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" 
                viewBox="0 0 {width} {height}" 
                preserveAspectRatio="xMidYMid meet">
    <g transform="{transform}">
        <path d="{path_cmd}" fill="black"/>
    </g>
    </svg>"""
    
    if debug_mode:
        logger.debug(f"Generated SVG:\n{svg}")
    
    return svg


def vector_svg_to_image(svg_str: str, target_w: int, target_h: int) -> Image.Image:
    """Fixed version: Render SVG to high-resolution PNG"""
    if not CAIROSVG_AVAILABLE:
        raise RuntimeError("cairosvg is not installed, cannot use vector mode. Please install first with `pip install cairosvg`")
    
    try:
        # Render to larger size
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=target_w,
            output_height=target_h,
            background_color="white"
        )
        img = Image.open(io.BytesIO(png_bytes))
        
        # Convert to grayscale image
        if img.mode != "L":
            img = img.convert("L")
            
        return img
    except Exception as e:
        logger.error(f"SVG rendering failed: {e}")
        raise


# =========================================================
# Main process: Batch export PNG from TTF (optional SVG export)
# =========================================================
def extract_font_images(ttf_path, output_dir, image_size, progress_bar, status_label,
                        mode, render_scale, threshold, rotate_mode, use_otsu, margin_ratio,
                        export_svg, keep_gray, debug_mode=False):
    """
    ttf_path: Font path
    mode: 'raster' | 'vector'
    render_scale: Supersampling multiple (effective in raster mode; used for initial rasterization resolution in vector mode)
    debug_mode: Whether to enable debug mode
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "meta.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["char", "unicode", "filename"])

        if mode == "vector":
            # Detailed dependency check
            missing_deps = []
            if not FONTTOOLS_AVAILABLE:
                missing_deps.append("fontTools")
            if not CAIROSVG_AVAILABLE:
                missing_deps.append("cairosvg")
                
            if missing_deps:
                msg = f"Vector mode requires the following libraries: {', '.join(missing_deps)}\nAutomatically switched to raster mode."
                messagebox.showwarning("Missing Dependencies", msg)
                mode = "raster"
                logger.warning(msg)
            else:
                logger.info("Vector mode dependency check passed")

        # TTFont is only needed in vector mode
        ttfont = None
        if mode == "vector":
            try:
                ttfont = TTFont(ttf_path)
                logger.info(f"Successfully loaded font: {ttf_path}")
            except Exception as e:
                messagebox.showerror("Error", f"TTFont opening failed: {e}\nWill automatically switch to raster mode.")
                mode = "raster"
                logger.error(f"Font loading failed: {e}")

        total = UNICODE_END - UNICODE_START + 1
        done = 0

        for code in range(UNICODE_START, UNICODE_END + 1):
            ch = chr(code)
            unicode_str = f"U+{code:04X}"
            filename = f"{unicode_str}.png"
            png_path = os.path.join(output_dir, filename)

            try:
                if mode == "raster":
                    logger.debug(f"Raster mode rendering: {unicode_str}")
                    im = raster_render_char(
                        ch, font_path=ttf_path, out_size=image_size, scale=render_scale,
                        threshold=threshold, rotate_mode=rotate_mode, use_otsu=use_otsu,
                        margin_ratio=margin_ratio, keep_gray=keep_gray
                    )
                    if im is None:
                        raise RuntimeError("Rendering failed or glyph is empty.")
                    im.save(png_path)

                else:  # vector
                    logger.debug(f"Vector mode rendering: {unicode_str}")
                    svg = glyph_to_svg(ttfont, ch, debug_mode)
                    if svg is None:
                        raise RuntimeError("Glyph does not exist or conversion failed.")
                        
                    # Save debug SVG
                    if debug_mode:
                        svg_path = os.path.join(output_dir, f"DEBUG_{unicode_str}.svg")
                        with open(svg_path, "w", encoding="utf-8") as fsvg:
                            fsvg.write(svg)
                        logger.info(f"Saved debug SVG: {svg_path}")

                    # First render to larger canvas, then smooth crop/align
                    render_size = max(512, image_size * max(2, int(render_scale)))
                    logger.debug(f"Render size: {render_size}x{render_size}")
                    
                    big = vector_svg_to_image(svg, render_size, render_size)
                    
                    # High-quality downsampling
                    if render_size > image_size:
                        big = big.resize((image_size, image_size), Image.LANCZOS)
                        logger.debug(f"Downsampled to target size: {image_size}x{image_size}")
                    
                    im_np = np.array(big)

                    if rotate_mode == 'cw90':
                        im_np = cv2.rotate(im_np, cv2.ROTATE_90_CLOCKWISE)
                    elif rotate_mode == 'ccw90':
                        im_np = cv2.rotate(im_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    centered = trim_center_square_smooth(im_np, out_size=image_size, margin_ratio=margin_ratio)
                    if centered is None:
                        raise RuntimeError("Cropping failed.")

                    centered = recenter_by_centroid_smart(centered, keep_gray=keep_gray,
                                                          threshold_val=threshold, use_otsu=use_otsu)

                    # If touching border, try increasing margin and retry 2 times
                    attempt = 0
                    m = margin_ratio
                    while touches_border(centered, margin_pixels=0) and attempt < 2:
                        attempt += 1
                        m = min(0.3, m + 0.02)
                        centered = trim_center_square_smooth(im_np, out_size=image_size, margin_ratio=m)
                        centered = recenter_by_centroid_smart(centered, keep_gray=keep_gray,
                                                              threshold_val=threshold, use_otsu=use_otsu)

                    Image.fromarray(centered).save(png_path)

                    if export_svg:
                        svg_path = os.path.join(output_dir, f"{unicode_str}.svg")
                        with open(svg_path, "w", encoding="utf-8") as fsvg:
                            fsvg.write(svg)

                writer.writerow([ch, unicode_str, filename])

            except Exception as e:
                error_msg = f"ERROR: {e}"
                logger.error(f"Failed to process character {unicode_str}: {e}")
                writer.writerow([ch, unicode_str, error_msg])

            # Progress feedback
            done += 1
            if progress_bar is not None:
                progress_bar["value"] = done / total * 100
                progress_bar.update_idletasks()
            if status_label is not None:
                status_label.config(text=f"Progress: {done}/{total}  Current: {unicode_str}")

    if status_label is not None:
        status_label.config(text=f"Completed: Output to {output_dir}")


# =========================================================
# GUI
# =========================================================
def run_gui():
    root = tk.Tk()
    root.title("Mongolian Font Extractor (Smooth Edge Version)")
    root.geometry("800x520")  # Adjusted height
    root.configure(bg="white")

    # --- Variables ---
    ttf_path_var   = tk.StringVar()
    out_dir_var    = tk.StringVar()
    img_size_var   = tk.IntVar(value=128)
    render_scale_var = tk.IntVar(value=6)  # Supersampling multiple (used in both raster/vector modes)
    threshold_var  = tk.IntVar(value=200)
    rotate_mode_var = tk.StringVar(value="none")
    use_otsu_var   = tk.BooleanVar(value=True)
    margin_ratio_var = tk.DoubleVar(value=0.12)
    mode_var       = tk.StringVar(value="raster")
    export_svg_var = tk.BooleanVar(value=False)
    keep_gray_var  = tk.BooleanVar(value=True)

    # --- Actions ---
    def choose_ttf():
        f = filedialog.askopenfilename(title="Select TTF/OTF", filetypes=[("Font", "*.ttf *.otf *.ttc"), ("All", "*.*")])
        if f:
            ttf_path_var.set(f)

    def choose_outdir():
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            out_dir_var.set(d)

    def open_output_folder():
        d = out_dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Information", "Please select or generate an output directory first.")
            return
        try:
            os.startfile(d)
        except Exception:
            messagebox.showinfo("Information", f"Output directory: {d}")

    # --- UI Layout ---
    pad = {"padx": 10, "pady": 6}

    frm_paths = tk.LabelFrame(root, text="Paths", bg="white")
    frm_paths.pack(fill="x", **pad)

    tk.Label(frm_paths, text="Font file:", bg="white").grid(row=0, column=0, sticky="e")
    tk.Entry(frm_paths, textvariable=ttf_path_var, width=60).grid(row=0, column=1, sticky="we")
    ttk.Button(frm_paths, text="Browse...", command=choose_ttf).grid(row=0, column=2, sticky="w", padx=6)

    tk.Label(frm_paths, text="Output folder:", bg="white").grid(row=1, column=0, sticky="e")
    tk.Entry(frm_paths, textvariable=out_dir_var, width=60).grid(row=1, column=1, sticky="we")
    ttk.Button(frm_paths, text="Select...", command=choose_outdir).grid(row=1, column=2, sticky="w", padx=6)

    frm_opts = tk.LabelFrame(root, text="Parameters", bg="white")
    frm_opts.pack(fill="x",** pad)

    # First row of parameters
    tk.Label(frm_opts, text="Output size:", bg="white").grid(row=0, column=0, sticky="e")
    tk.Spinbox(frm_opts, from_=32, to=1024, textvariable=img_size_var, width=6).grid(row=0, column=1, sticky="w")

    tk.Label(frm_opts, text="Supersampling:", bg="white").grid(row=0, column=2, sticky="e")
    tk.Spinbox(frm_opts, from_=2, to=12, textvariable=render_scale_var, width=6).grid(row=0, column=3, sticky="w")

    tk.Label(frm_opts, text="Threshold:", bg="white").grid(row=0, column=4, sticky="e")
    tk.Spinbox(frm_opts, from_=0, to=255, textvariable=threshold_var, width=6).grid(row=0, column=5, sticky="w")
    ttk.Checkbutton(frm_opts, text="Otsu", variable=use_otsu_var).grid(row=0, column=6, sticky="w")

    # Second row of parameters
    tk.Label(frm_opts, text="Margin ratio:", bg="white").grid(row=1, column=0, sticky="e")
    tk.Spinbox(frm_opts, from_=0.00, to=0.40, increment=0.01,
               textvariable=margin_ratio_var, width=6).grid(row=1, column=1, sticky="w")

    tk.Label(frm_opts, text="Rotation:", bg="white").grid(row=1, column=2, sticky="e")
    ttk.Combobox(frm_opts, textvariable=rotate_mode_var, values=["none", "cw90", "ccw90"],
                 width=8, state="readonly").grid(row=1, column=3, sticky="w")

    tk.Label(frm_opts, text="Mode:", bg="white").grid(row=1, column=4, sticky="e")
    ttk.Combobox(frm_opts, textvariable=mode_var, values=["raster", "vector"],
                 width=8, state="readonly").grid(row=1, column=5, sticky="w")

    ttk.Checkbutton(frm_opts, text="Export SVG (vector mode)", variable=export_svg_var).grid(row=1, column=6, sticky="w")

    # Third row of parameters
    ttk.Checkbutton(frm_opts, text="Preserve grayscale edges (smoother)", variable=keep_gray_var).grid(row=2, column=0, columnspan=2, sticky="w")

    # Progress
    frm_prog = tk.LabelFrame(root, text="Progress", bg="white")
    frm_prog.pack(fill="x", **pad)
    progress = ttk.Progressbar(frm_prog, orient="horizontal", length=520, mode="determinate")
    progress.pack(padx=10, pady=6)
    status_label = tk.Label(frm_prog, text="Ready", anchor="w", bg="white")
    status_label.pack(fill="x", padx=10, pady=(0,6))

    def on_start():
        ttf_path = ttf_path_var.get().strip()
        out_dir = out_dir_var.get().strip()
        if not ttf_path or not os.path.isfile(ttf_path):
            messagebox.showerror("Error", "Please select a valid TTF font file.")
            return
        if not out_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        # Disable start button to prevent duplicate clicks
        btn_start.config(state="disabled")
        progress["value"] = 0
        
        # Execute extraction in a new thread to avoid GUI freezing
        import threading
        thread = threading.Thread(target=extract_font_images, args=(
            ttf_path, out_dir, img_size_var.get(), progress, status_label,
            mode_var.get(), render_scale_var.get(), threshold_var.get(),
            rotate_mode_var.get(), use_otsu_var.get(), margin_ratio_var.get(),
            export_svg_var.get(), keep_gray_var.get(), False  # debug_mode is always False
        ))
        thread.daemon = True
        thread.start()
        
        # Check thread status periodically
        def check_thread():
            if thread.is_alive():
                root.after(500, check_thread)
            else:
                btn_start.config(state="normal")
        
        root.after(500, check_thread)

    # Button area
    frm_btns = tk.Frame(root, bg="white")
    frm_btns.pack(fill="x",** pad)
    btn_start = ttk.Button(frm_btns, text="Start Extraction", command=on_start)
    btn_start.pack(side="left", padx=10)
    ttk.Button(frm_btns, text="Open Output Folder", command=open_output_folder).pack(side="left", padx=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()