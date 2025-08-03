import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# 字符范围：回鹘蒙文 U+1820 ~ U+1842
UNICODE_START = 0x1820
UNICODE_END = 0x1842

def render_char_image(ch, font, render_size, image_size, threshold, rotate):
    img = Image.new("L", (render_size, render_size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
    except:
        return None
    if not bbox or bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
        return None
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (render_size - w) // 2 - bbox[0]
    y = (render_size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    img_np = np.array(img)
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    if rotate:
        binary = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
    small = cv2.resize(binary, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(small)

def extract_mongol_font(ttf_path, output_dir, font_size, image_size, progress_bar, threshold=220, rotate=True):
    if not os.path.exists(ttf_path):
        messagebox.showerror("错误", "字体文件不存在")
        return
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.truetype(ttf_path, font_size)
    render_size = 512
    total = UNICODE_END - UNICODE_START + 1
    count = 0
    for i, code in enumerate(range(UNICODE_START, UNICODE_END + 1)):
        ch = chr(code)
        img = render_char_image(ch, font, render_size, image_size, threshold, rotate)
        if img:
            img.save(os.path.join(output_dir, f"u{code:04x}.png"))
            count += 1
        progress_bar["value"] = (i + 1) / total * 100
        root.update_idletasks()
    messagebox.showinfo("完成", f"提取成功！共生成 {count} 张图像，保存在：\n{output_dir}")

def open_output_folder():
    path = output_entry.get()
    if os.path.exists(path):
        os.startfile(path)
    else:
        messagebox.showwarning("提示", "输出文件夹不存在")

# GUI 界面初始化
root = tk.Tk()
root.title("回鹘蒙文字体图像提取工具")
root.geometry("640x600")
root.configure(bg="#f0f4f8")

# 标题栏
header = tk.Frame(root, bg="#165DFF", height=60)
header.pack(fill=tk.X)
header_label = tk.Label(header, text="回鹘蒙文字体图像提取工具", font=("Microsoft YaHei", 18, "bold"), bg="#165DFF", fg="white")
header_label.pack(pady=10, padx=20, anchor="w")

# 主白色区域
main_card = tk.Frame(root, bg="white", bd=1, relief="solid")
main_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="white", font=("Microsoft YaHei", 10))
style.configure("TEntry", padding=6)
style.configure("Primary.TButton", background="#165DFF", foreground="white", font=("Microsoft YaHei", 10, "bold"))
style.configure("Horizontal.TProgressbar", background="#165DFF", troughcolor="#edf2f7")

def labeled_entry(master, label_text, default=""):
    frame = tk.Frame(master, bg="white")
    frame.pack(fill=tk.X, padx=15, pady=8)
    label = ttk.Label(frame, text=label_text, width=16)
    label.grid(row=0, column=0, sticky=tk.W)
    entry = ttk.Entry(frame)
    entry.insert(0, default)
    entry.grid(row=0, column=1, sticky=tk.EW)
    frame.columnconfigure(1, weight=1)
    return entry

font_entry = labeled_entry(main_card, "字体文件路径：")
ttk.Button(main_card, text="浏览字体", command=lambda: font_entry.delete(0, tk.END) or font_entry.insert(0, filedialog.askopenfilename(filetypes=[("字体文件", "*.ttf")]))).pack(padx=15, anchor="w")

output_entry = labeled_entry(main_card, "输出文件夹路径：", default="output")
ttk.Button(main_card, text="选择输出目录", command=lambda: output_entry.delete(0, tk.END) or output_entry.insert(0, filedialog.askdirectory())).pack(padx=15, anchor="w")

fontsize_entry = labeled_entry(main_card, "字体大小：", default="360")
imagesize_entry = labeled_entry(main_card, "图像尺寸：", default="128")

rotate_var = tk.BooleanVar(value=True)
rotate_check = tk.Checkbutton(main_card, text="顺时针旋转90°（竖排）", variable=rotate_var, font=("Microsoft YaHei", 10), bg="white")
rotate_check.pack(anchor="w", padx=15, pady=8)

progress = ttk.Progressbar(main_card, orient="horizontal", mode="determinate")
progress.pack(fill=tk.X, padx=15, pady=(5, 10))

status_label = tk.Label(main_card, text="状态：就绪", font=("Microsoft YaHei", 9), bg="white", fg="#666")
status_label.pack(anchor="w", padx=15)

btn_frame = tk.Frame(root, bg="#f0f4f8")
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="开始提取", style="Primary.TButton", command=lambda: extract_mongol_font(font_entry.get(), output_entry.get(), int(fontsize_entry.get()), int(imagesize_entry.get()), progress, rotate=rotate_var.get())).pack(side=tk.LEFT, padx=10)
ttk.Button(btn_frame, text="打开输出文件夹", command=open_output_folder).pack(side=tk.LEFT, padx=10)

root.mainloop()
