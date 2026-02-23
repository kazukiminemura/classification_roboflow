import argparse
import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI tester for YOLO26n-cls classification model")
    parser.add_argument(
        "--model",
        default="artifacts/yolo26n_cls_finetune/weights/best.pt",
        help="Path to trained model (.pt)",
    )
    parser.add_argument("--img-size", default=224, type=int, help="Inference image size")
    parser.add_argument("--topk", default=5, type=int, help="How many classes to show")
    return parser.parse_args()


class ClassifierGUI:
    def __init__(self, root: tk.Tk, model_path: str, img_size: int, topk: int):
        self.root = root
        self.root.title("YOLO26n-cls GUI Tester")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.img_size = img_size
        self.topk = topk
        self.model = None
        self.current_image = None
        self.video_cap = None
        self.video_running = False
        self.video_after_id = None
        self.video_delay_ms = 33

        self.model_var = tk.StringVar(value=model_path)
        self.media_var = tk.StringVar(value="")
        self.top1_var = tk.StringVar(value="Top1: -")
        self.status_var = tk.StringVar(value="Status: idle")

        self._build_ui()
        self._autoload_model()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        model_frame = ttk.LabelFrame(main, text="Model", padding=10)
        model_frame.pack(fill=tk.X)
        ttk.Entry(model_frame, textvariable=self.model_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT, padx=6)
        ttk.Button(model_frame, text="Load", command=self.load_model).pack(side=tk.LEFT)

        media_frame = ttk.LabelFrame(main, text="Input (Image / Video)", padding=10)
        media_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Entry(media_frame, textvariable=self.media_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(media_frame, text="Browse", command=self.browse_media).pack(side=tk.LEFT, padx=6)
        ttk.Button(media_frame, text="Predict", command=self.predict_media).pack(side=tk.LEFT, padx=6)
        ttk.Button(media_frame, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=6)
        ttk.Button(media_frame, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT)

        result_frame = ttk.LabelFrame(main, text="Prediction", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(result_frame, textvariable=self.top1_var, font=("", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor=tk.W, pady=(2, 0))

        columns = ("class", "confidence")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=8)
        self.tree.heading("class", text="Class")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("class", width=300)
        self.tree.column("confidence", width=120, anchor=tk.E)
        self.tree.pack(fill=tk.X, pady=(6, 10))

        self.image_label = ttk.Label(result_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def _autoload_model(self):
        path = pathlib.Path(self.model_var.get())
        if path.exists():
            self.load_model()

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def browse_media(self):
        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.stop_video(silent=True)
            self.media_var.set(path)
            if self.is_video(pathlib.Path(path)):
                self.status_var.set("Status: video selected")
            else:
                self.status_var.set("Status: image selected")
                self.show_pil_image(Image.open(path).convert("RGB"))

    def load_model(self):
        model_path = pathlib.Path(self.model_var.get())
        if not model_path.exists():
            messagebox.showerror("Model Error", f"Model not found:\n{model_path}")
            return
        try:
            self.model = YOLO(str(model_path))
            messagebox.showinfo("Model", f"Loaded model:\n{model_path}")
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))

    def is_video(self, path: pathlib.Path) -> bool:
        return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

    def show_pil_image(self, image: Image.Image):
        image.thumbnail((850, 420))
        self.current_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_image)

    def _update_prediction(self, result):
        probs = result.probs
        if probs is None:
            raise RuntimeError("No classification probability found in result.")
        names = result.names
        prob_values = probs.data.tolist()
        ranked = sorted(enumerate(prob_values), key=lambda x: x[1], reverse=True)[: self.topk]
        self.top1_var.set(f"Top1: {names[probs.top1]} ({float(probs.top1conf):.4f})")
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, conf in ranked:
            self.tree.insert("", tk.END, values=(names[idx], f"{conf:.4f}"))

    def predict_media(self):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return

        media_path = pathlib.Path(self.media_var.get())
        if not media_path.exists():
            messagebox.showerror("Input Error", f"File not found:\n{media_path}")
            return

        try:
            self.stop_video(silent=True)
            if self.is_video(media_path):
                cap = cv2.VideoCapture(str(media_path))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError("Could not read first frame from video.")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model.predict(source=frame_rgb, imgsz=self.img_size, verbose=False)
                self._update_prediction(results[0])
                self.show_pil_image(Image.fromarray(frame_rgb))
                self.status_var.set("Status: predicted first video frame")
            else:
                results = self.model.predict(source=str(media_path), imgsz=self.img_size, verbose=False)
                self._update_prediction(results[0])
                self.show_pil_image(Image.open(media_path).convert("RGB"))
                self.status_var.set("Status: image predicted")
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def start_video(self):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return

        media_path = pathlib.Path(self.media_var.get())
        if not media_path.exists():
            messagebox.showerror("Input Error", f"File not found:\n{media_path}")
            return
        if not self.is_video(media_path):
            messagebox.showerror("Input Error", "Selected file is not a video.")
            return

        self.stop_video(silent=True)
        self.video_cap = cv2.VideoCapture(str(media_path))
        if not self.video_cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video:\n{media_path}")
            self.video_cap = None
            return

        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.video_delay_ms = max(15, int(1000 / fps))
        else:
            self.video_delay_ms = 33

        self.video_running = True
        self.status_var.set("Status: video running")
        self._process_video_frame()

    def _process_video_frame(self):
        if not self.video_running or self.video_cap is None:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.stop_video(silent=True)
            self.status_var.set("Status: video finished")
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.predict(source=frame_rgb, imgsz=self.img_size, verbose=False)
            self._update_prediction(results[0])
            self.show_pil_image(Image.fromarray(frame_rgb))
        except Exception as exc:
            self.stop_video(silent=True)
            messagebox.showerror("Prediction Error", str(exc))
            return

        self.video_after_id = self.root.after(self.video_delay_ms, self._process_video_frame)

    def stop_video(self, silent: bool = False):
        self.video_running = False
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        if not silent:
            self.status_var.set("Status: video stopped")

    def on_close(self):
        self.stop_video(silent=True)
        self.root.destroy()


def main():
    args = parse_args()
    root = tk.Tk()
    ClassifierGUI(root, args.model, args.img_size, args.topk)
    root.mainloop()


if __name__ == "__main__":
    main()
