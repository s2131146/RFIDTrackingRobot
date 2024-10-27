from tkinter import font
import cv2
import tqueue
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext as st
from PIL import Image, ImageTk
from tracker import Tracker

DEF_MARGIN = 12
STICKY_UP = "n"
STICKY_CENTER = "nsew"
STICKY_LEFT = "w"
STICKY_RIGHT = "e"


class GUI:
    queue = tqueue.TQueue()
    timeline_index = 0

    def __init__(self, _tracker: Tracker, root: tk.Tk):
        self.tracker = _tracker
        self.root = root
        self.root.title("Metoki - RFID Tracking Robot")
        self.root.bind("<KeyPress>", self.key_pressed)
        self.root.bind("<Button-1>", self.on_click)

        self.custom_font = font.Font(family="Consolas", size=10)
        self.bold_font = font.Font(family="Verdana Bold", size=18, weight="bold")

        # モーター速度表示用のStringVarを初期化
        self.motor_left_var = tk.StringVar(value="0%")
        self.motor_right_var = tk.StringVar(value="0%")

        # モード選択用のStringVarを初期化
        self.mode_var = tk.StringVar(value=Tracker.Mode.CAM_ONLY.name)

        # 自動スクロールの変数を初期化
        self.var_auto_scroll_received = tk.IntVar(value=1)
        self.var_auto_scroll_sent = tk.IntVar(value=1)

        self.create_main_frame()
        self.create_control_frame()

        self.update_stop()

        self.queue.add("init")

    def create_main_frame(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(
            row=0,
            column=0,
            padx=(DEF_MARGIN, 0),
            pady=(DEF_MARGIN, 0),
            sticky=STICKY_UP,
        )

        self.video_frame = tk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0, padx=0, pady=0)

        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.grid(
            row=1, column=0, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_CENTER
        )

        # カラムの重み付けをすべて1に設定
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_columnconfigure(2, weight=1)

        self.var_enable_tracking = tk.IntVar(value=1)
        self.enable_tracking_checkbox = tk.Checkbutton(
            self.bottom_frame, text="Enable tracking", variable=self.var_enable_tracking
        )
        self.enable_tracking_checkbox.grid(
            row=0, column=0, padx=0, pady=0, sticky=STICKY_LEFT
        )

        self.label_wheel = tk.Label(self.bottom_frame)
        self.label_wheel.grid(row=0, column=1, padx=0, pady=0, sticky=STICKY_CENTER)

        # RFIDアンテナのアイコンを「Enable tracking」の右側に固定
        self.rfid_frame = tk.Frame(self.bottom_frame)
        self.rfid_frame.grid(row=1, column=0, padx=(0, 0), pady=0, sticky=STICKY_CENTER)

        self.rfid_values = [tk.StringVar(value="0") for _ in range(4)]
        self.rfid_canvases = []
        self.rfid_text_ids = []
        self.rfid_rect_ids = []  # 四角形のIDを保存するリスト

        rfid_labels = ["CENTER", "LEFT", "RIGHT", "REAR"]

        for i in range(4):
            canvas = tk.Canvas(
                self.rfid_frame, width=50, height=50, highlightthickness=0
            )
            canvas.grid(row=0, column=i, padx=5, pady=0)

            rect_id = self.draw_rounded_rectangle(
                canvas, 5, 5, 45, 45, radius=10, fill="white", outline="black"
            )
            self.rfid_rect_ids.append(rect_id)

            text_id = canvas.create_text(
                25, 25, text=self.rfid_values[i].get(), font=self.custom_font
            )
            self.rfid_text_ids.append(text_id)

            self.rfid_values[i].trace_add(
                "write", lambda *args, index=i: self.update_canvas_text(index, "rfid")
            )

            self.rfid_canvases.append(canvas)

            label = tk.Label(
                self.rfid_frame, text=rfid_labels[i], font=self.custom_font
            )
            label.grid(row=1, column=i, padx=0, pady=(0, 0), sticky=STICKY_CENTER)

        # モーター表示用のフレームを追加
        self.motor_frame = tk.Frame(self.bottom_frame)
        self.motor_frame.grid(
            row=1,
            column=1,
            columnspan=3,
            padx=0,
            pady=(10, 0),
            sticky=STICKY_CENTER,
        )

        # motor_frame 内のカラムを設定
        self.motor_frame.grid_columnconfigure(0, weight=1)
        self.motor_frame.grid_columnconfigure(1, weight=1)
        self.motor_frame.grid_columnconfigure(2, weight=1)

        self.motor_values = [self.motor_left_var, self.motor_right_var]
        self.motor_canvases = []
        self.motor_text_ids = []
        self.motor_rect_ids = []

        motor_labels = ["L Motor", "R Motor"]

        for i in range(2):
            # motor_frame 内での位置を調整
            canvas = tk.Canvas(
                self.motor_frame, width=50, height=50, highlightthickness=0
            )
            canvas.grid(
                row=0, column=i, padx=20, pady=0  # 中央に配置するための padx を増やす
            )

            rect_id = self.draw_rounded_rectangle(
                canvas, 5, 5, 45, 45, radius=10, fill="white", outline="black"
            )
            self.motor_rect_ids.append(rect_id)

            text_id = canvas.create_text(
                25, 25, text=self.motor_values[i].get(), font=self.custom_font
            )
            self.motor_text_ids.append(text_id)

            self.motor_values[i].trace_add(
                "write", lambda *args, index=i: self.update_canvas_text(index, "motor")
            )

            self.motor_canvases.append(canvas)

            label = tk.Label(
                self.motor_frame, text=motor_labels[i], font=self.custom_font
            )
            label.grid(
                row=1, column=i, padx=20, pady=(5, DEF_MARGIN), sticky=STICKY_CENTER
            )

    def draw_rounded_rectangle(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        points = [
            x1 + radius,
            y1,
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    def update_canvas_text(self, index, key):
        if key == "rfid":
            canvas = self.rfid_canvases[index]
            text_id = self.rfid_text_ids[index]
            new_text = self.rfid_values[index].get()
        elif key == "motor":
            canvas = self.motor_canvases[index]
            text_id = self.motor_text_ids[index]
            new_text = self.motor_values[index].get()
            canvas.itemconfig(self.motor_rect_ids[index], fill="white")
        else:
            return

        canvas.itemconfig(text_id, text=new_text)

    def create_control_frame(self):
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(
            row=0, column=1, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_UP
        )

        self.command_frame = tk.Frame(self.control_frame)
        self.command_frame.grid(
            row=0, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_UP
        )

        self.create_command_entry()
        self.create_scrolled_text()
        self.create_status_frame()

    def create_command_entry(self):
        # Mode選択用のラベルとドロップダウンメニューを追加
        self.mode_label = tk.Label(self.command_frame, text="Mode:")
        self.mode_label.grid(row=0, column=0, padx=0, pady=(0, 5), sticky=STICKY_LEFT)

        self.mode_options = ["CAM_ONLY", "DUAL", "RFID_ONLY"]
        self.mode_menu = ttk.OptionMenu(
            self.command_frame,
            self.mode_var,
            self.mode_var.get(),
            *self.mode_options,
            command=self.mode_selected,  # 選択時に呼び出されるメソッド
        )
        self.mode_menu.grid(
            row=0, column=1, padx=DEF_MARGIN, pady=(0, 5), sticky=STICKY_LEFT
        )

        # Commandラベルとエントリーの位置を下にずらす
        self.command_label = tk.Label(self.command_frame, text="Command:")
        self.command_label.grid(row=1, column=0, padx=0, pady=0, sticky=STICKY_LEFT)

        self.command_entry = tk.Entry(self.command_frame, width=25)
        self.command_entry.grid(row=1, column=1, padx=DEF_MARGIN, pady=5)
        self.command_entry.bind("<Return>", self.command_enter_pressed)
        self.command_entry.focus_set()

        self.send_button = ttk.Button(
            self.command_frame, text="Send", command=self.send_command
        )
        self.send_button.grid(row=1, column=2, padx=0, pady=5)

    def mode_selected(self, selected_mode):
        """モードが選択された際に呼び出されるメソッド"""
        self.update_status(f"Mode selected: {selected_mode}")
        self.tracker.update_mode()

    def create_scrolled_text(self):
        # 受信テキストと送信テキストのスコールドテキストウィジェットを作成
        self.received_text = self.create_scrolled_widget(
            self.control_frame,
            "AutoScroll",
            self.var_auto_scroll_received,
            row=1,
        )
        self.label_received = tk.Label(
            self.control_frame,
            text="Initialized.",
            anchor="w",
            justify="left",
            font=self.custom_font,
        )
        self.label_received.grid(
            row=3, column=0, padx=DEF_MARGIN, pady=5, sticky=STICKY_LEFT
        )

        self.sent_text = self.create_scrolled_widget(
            self.control_frame, "AutoScroll", self.var_auto_scroll_sent, row=4
        )
        self.label_sent = tk.Label(
            self.control_frame,
            text="",
            anchor="w",
            justify="left",
            font=self.custom_font,
        )
        self.label_sent.grid(
            row=7, column=0, padx=DEF_MARGIN, pady=5, sticky=STICKY_LEFT
        )

    def create_scrolled_widget(self, frame, label_text, var_auto_scroll, row):
        scrolled_widget = st.ScrolledText(
            frame, wrap=tk.WORD, width=40, height=10, font=self.custom_font
        )
        scrolled_widget.grid(row=row, column=0, padx=0, pady=(5, 0))
        auto_scroll_checkbox = tk.Checkbutton(
            frame, text=label_text, variable=var_auto_scroll
        )
        auto_scroll_checkbox.grid(
            row=row + 1, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_LEFT
        )
        return scrolled_widget

    def create_status_frame(self):
        self.status_frame = tk.Frame(self.control_frame)
        self.status_frame.grid(
            row=8, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_CENTER
        )
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_frame.grid_columnconfigure(1, weight=1)

        self.label_status = tk.Label(
            self.status_frame,
            text="Ready",
            font=self.custom_font,
            anchor="w",
            justify="left",
        )
        self.label_status.grid(row=0, column=0, padx=0, pady=0, sticky=STICKY_LEFT)

        self.label_seg = tk.Label(self.status_frame, text="0", font=self.custom_font)
        self.label_seg.grid(row=0, column=1, padx=0, pady=0, sticky=STICKY_RIGHT)

    def on_click(self, e):
        if not isinstance(e.widget, tk.Entry):
            self.root.focus_set()

    def key_pressed(self, e):
        focus = self.root.focus_get()
        if focus is not None and focus != self.root:
            return

        if e.keysym == "x":
            self.command_stop_start()
        elif e.keysym == "q":
            self.__del__()

    def command_enter_pressed(self, e):
        self.send_command()

    def send_command(self):
        command = self.command_entry.get()
        res = self.tracker.send(command)

        if res:
            self.update_status("Sent: " + command.upper())
        else:
            self.update_status("Invalid Command: " + command.upper())

        self.command_entry.delete(0, tk.END)

    def update_stop(self, connected=True):
        if self.tracker.stop:
            self.update_status("Motor stopped.")
            text, bg_color, fg_color = (
                "START",
                "white",
                "green" if connected else "gray",
            )
        else:
            self.update_status("Motor started.")
            text, bg_color, fg_color = "STOP", "yellow", "red"

        self.stop_button = tk.Button(
            self.control_frame,
            text=text,
            font=self.bold_font,
            borderwidth=2,
            bg=bg_color,
            fg=fg_color,
            activebackground=bg_color,
            activeforeground=fg_color,
            relief="solid",
            command=self.command_stop_start,
        )
        self.stop_button.grid(
            row=6, column=0, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_CENTER
        )
        self.stop_button.config(padx=20, pady=5)

        if not connected:
            self.root.configure(bg="red")
        else:
            self.root.configure(bg="SystemButtonFace")

    def command_stop_start(self):
        if self.tracker.stop:
            self.tracker.start_motor()
        else:
            self.tracker.stop_motor()

    def update_status(self, status):
        self.label_status.config(text=status)

    def update_wheel(self):
        if self.queue.has("g"):
            self.label_wheel.config(text=self.queue.get("g"))

    def update_seg(self, seg):
        self.label_seg.config(text=seg)

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if not hasattr(self, "imgtk"):
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.config(image=self.imgtk)
        else:
            self.imgtk.paste(img)

        self.update_commands("s", self.sent_text, self.label_sent)
        self.update_commands("r", self.received_text, self.label_received)
        self.update_wheel()

    def update_commands(self, key, text_widget, label_widget):
        if self.queue.has(key):
            cmd = self.queue.get(key)
            self.insert_text(text_widget, cmd, key)
            label_widget.config(text=cmd)

    def insert_text(self, text_widget, cmd, key):
        if not text_widget.get("1.0", tk.END).strip():
            text_widget.insert(tk.END, "{}: {}".format(self.timeline_index, cmd))
        else:
            text_widget.insert(tk.END, "\n{}: {}".format(self.timeline_index, cmd))

        self.timeline_index += 1

        if key == "s" and self.var_auto_scroll_sent.get():
            text_widget.see(tk.END)
        elif key == "r" and self.var_auto_scroll_received.get():
            text_widget.see(tk.END)

    def update_rfid_values(self, counts):
        """各アンテナの検出回数を更新します。

        Args:
            counts (dict): 各アンテナの検出回数
        """
        # アンテナの順序に基づいて更新
        antenna_order = ["CENTER", "LEFT", "RIGHT", "REAR"]
        for i, antenna in enumerate(antenna_order):
            count = counts.get(i + 1, 0)
            self.rfid_values[i].set(str(count))
            canvas = self.rfid_canvases[i]
            rect_id = self.rfid_rect_ids[i]

            # 検出回数に応じて色を変更
            if count == 0:
                color = "white"
            else:
                max_count = 12
                green_intensity = min(255, int((count / max_count) * 255))
                color = f"#{255-green_intensity:02x}{255:02x}{255-green_intensity:02x}"

            canvas.itemconfig(rect_id, fill=color)

    def update_motor_values(self, left_value, right_value):
        """モーターの速度を更新します。

        Args:
            left_value (str): 左モーターの速度
            right_value (str): 右モーターの速度
        """
        l_str = str(left_value) + "%"
        r_str = str(right_value) + "%"
        self.motor_left_var.set(l_str)
        self.motor_right_var.set(r_str)

    def __del__(self):
        self.tracker.close()
