import os
import tkinter as tk
from tkinter import filedialog, Canvas, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np

class ImageAnnotator:
    def __init__(self, master):
        self.master = master
        master.title("Image Annotator")

        # 初始化变量
        self.image_dir = None
        self.image_files = []
        self.current_image_index = 0
        self.image = None
        self.photo = None
        self.canvas = None
        self.center_point = None
        self.p1 = None
        self.p2 = None
        self.line_id = None
        self.dot_size = 5

        # 创建 GUI 元素
        self.create_widgets()

    def create_widgets(self):
        # 文件夹选择按钮
        self.folder_button = tk.Button(self.master, text="选择文件夹", command=self.select_folder)
        self.folder_button.pack(pady=10)

        # 图片显示区域
        self.canvas = Canvas(self.master, width=800, height=600)
        self.canvas.pack()

        # 鼠标位置标签
        self.mouse_label = tk.Label(self.master, text="")
        self.mouse_label.pack()

        # 控制按钮
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=10)

        self.prev_button = tk.Button(self.button_frame, text="上一张", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.button_frame, text="下一张", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="清除", command=self.clear_annotations)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.finish_button = tk.Button(self.button_frame, text="完成", command=self.finish_annotation)
        self.finish_button.pack(side=tk.LEFT, padx=5)

        # 绑定鼠标事件
        self.canvas.bind("<Motion>", self.show_mouse_position)
        self.canvas.bind("<Button-1>", self.annotate_point)

    def select_folder(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.image_files.sort()
            self.current_image_index = 0
            self.load_image()

    def load_image(self):
        if self.image_files:
            image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
            self.image = Image.open(image_path)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # 读取质心点坐标
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    x1, y1 = map(int, f.readline().split())
                    self.center_point = (x1, y1)
                    self.draw_center_point()

    def draw_center_point(self):
        if self.center_point:
            x, y = self.center_point
            self.canvas.create_oval(x - self.dot_size, y - self.dot_size, x + self.dot_size, y + self.dot_size, fill="red", outline="red")

    def show_mouse_position(self, event):
        self.mouse_label.config(text=f"鼠标位置: x={event.x}, y={event.y}")

        if self.p1:
            # 显示辅助线
            if self.line_id:
                self.canvas.delete(self.line_id)
            self.line_id = self.canvas.create_line(self.p1[0], self.p1[1], event.x, event.y, fill="blue", dash=(4, 4))

    def annotate_point(self, event):
        if not self.p1:
            self.p1 = (event.x, event.y)
            self.canvas.create_oval(event.x - self.dot_size, event.y - self.dot_size, event.x + self.dot_size, event.y + self.dot_size, fill="green", outline="green")
        elif not self.p2:
            self.p2 = (event.x, event.y)
            self.canvas.create_oval(event.x - self.dot_size, event.y - self.dot_size, event.x + self.dot_size, event.y + self.dot_size, fill="blue", outline="blue")
            self.canvas.delete(self.line_id)  # 清除辅助线

    def clear_annotations(self):
        if self.p1 or self.p2:
            self.p1 = None
            self.p2 = None
            self.canvas.delete("all")
            self.load_image()

    def finish_annotation(self):
        if self.p1 and self.p2:
            # 计算 dx, dy, R
            x1, y1 = self.center_point
            dx = self.p1[0] - x1
            dy = self.p1[1] - y1
            R = self.p1[0] - self.p2[0]

            # 保存标注结果
            image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
            annotation_path = os.path.splitext(image_path)[0] + "_annotation.txt"
            with open(annotation_path, 'w') as f:
                f.write(f"{dx} {dy} {R}\n")

            # 切换到下一张图片
            self.next_image()
        else:
            messagebox.showwarning("警告", "请先标注两个点！")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.clear_annotations()
            self.load_image()

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.clear_annotations()
            self.load_image()

root = tk.Tk()
app = ImageAnnotator(root)
root.mainloop()