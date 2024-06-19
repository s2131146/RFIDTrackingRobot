import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV and Tkinter")

        # Video frame
        self.video_frame = tk.Label(root)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        # Right frame for controls
        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Text box
        self.text_box = tk.Text(self.control_frame, height=10, width=30)
        self.text_box.grid(row=0, column=0, padx=5, pady=5)

        # Button
        self.button = ttk.Button(self.control_frame, text="Print Text", command=self.print_text)
        self.button.grid(row=1, column=0, padx=5, pady=5)

        # Update Button
        self.update_button = ttk.Button(self.control_frame, text="Update Text", command=self.update_text)
        self.update_button.grid(row=2, column=0, padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(self.control_frame, text="Status: Ready", anchor="w", justify="left")
        self.status_label.grid(row=3, column=0, padx=5, pady=5)

        # Capture video from webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def print_text(self):
        text = self.text_box.get("1.0", tk.END).strip()
        messagebox.showinfo("Text", text)
        self.update_status("Printed text.")

    def update_text(self):
        # ここで必要な処理をして、テキストを更新する
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, "Updated text goes here.")
        self.update_status("Text has been updated.")

    def update_status(self, status):
        # 現在のステータステキストを取得し、新しいステータスと結合して更新
        current_status = self.status_label.cget("text")
        new_status = f"{status}" if current_status else status
        self.status_label.config(text=new_status)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.config(image=imgtk)
        self.root.after(10, self.update_frame)

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
