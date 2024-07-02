from tkinter import font
import cv2
import tqueue
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext as st
from PIL import Image, ImageTk

DEF_MARGIN = 12
STICKY_UP = "n"
STICKY_DOWN = "s"
STICKY_RIGHT = "e"
STICKY_LEFT = "w"
STICKY_CENTER = "nsew"
STICKY_CENTER_HORIZONTAL = "ew"


class App:
    queue = tqueue.TQueue()
    timeline_index = 0

    def __init__(self, _tracker, root):
        self.tracker = _tracker
        self.root = root
        self.root.title("Metoki - RFID Tracking Robot")
        self.root.bind("<KeyPress>", self.key_pressed)
        self.root.bind("<Button-1>", self.on_click)

        self.main_frame = tk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=(DEF_MARGIN, 0), pady=(DEF_MARGIN, 0), sticky=STICKY_UP)

        self.video_frame = tk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0, padx=0, pady=0)

        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.grid(row=1, column=0, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_CENTER)

        self.var_enable_tracking = tk.IntVar(value=1)
        self.enable_tracking_checkbox = tk.Checkbutton(self.bottom_frame, text="Enable tracking", variable=self.var_enable_tracking)
        self.enable_tracking_checkbox.grid(row=0, column=0, padx=0, pady=0, sticky=STICKY_LEFT)

        self.label_wheel = tk.Label(self.bottom_frame)
        self.label_wheel.grid(row=0, column=1, padx=0, pady=0, sticky=STICKY_CENTER_HORIZONTAL)

        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=0, column=1, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_UP)

        self.command_frame = tk.Frame(self.control_frame)
        self.command_frame.grid(row=0, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_UP)

        self.command_label = tk.Label(self.command_frame, text="Command:")
        self.command_label.grid(row=0, column=0, padx=0, pady=0, sticky=STICKY_LEFT)

        self.command_entry = tk.Entry(self.command_frame, width=25)
        self.command_entry.grid(row=0, column=1, padx=DEF_MARGIN, pady=5)
        self.command_entry.bind("<Return>", self.comamnd_enter_pressed)
        self.command_entry.focus_set()

        self.send_button = ttk.Button(self.command_frame, text="Send", command=self.send_command)
        self.send_button.grid(row=0, column=2, padx=0, pady=5)

        self.received_text = st.ScrolledText(self.control_frame, wrap=tk.WORD, width=40, height=10, font=("Consolas", 10))
        self.received_text.grid(row=1, column=0, padx=0, pady=(5, 0))

        self.var_auto_scroll_received = tk.IntVar(value=1)
        self.auto_scroll_received_checkbox = tk.Checkbutton(self.control_frame, text="AutoScroll", variable=self.var_auto_scroll_received)
        self.auto_scroll_received_checkbox.grid(row=2, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_LEFT)

        custom_font = font.Font(family="Consolas", size=10)
        self.label_received = tk.Label(self.control_frame, text="Initialized.", anchor="w", justify="left", font=custom_font)
        self.label_received.grid(row=3, column=0, padx=DEF_MARGIN, pady=5, sticky=STICKY_LEFT)

        self.sent_text = st.ScrolledText(self.control_frame, wrap=tk.WORD, width=40, height=10, font=("Consolas", 10))
        self.sent_text.grid(row=4, column=0, padx=0, pady=(5, 0))

        self.var_auto_scroll_sent = tk.IntVar(value=1)
        self.auto_scroll_sent_checkbox = tk.Checkbutton(self.control_frame, text="AutoScroll", variable=self.var_auto_scroll_sent)
        self.auto_scroll_sent_checkbox.grid(row=5, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_LEFT)

        self.label_sent = tk.Label(self.control_frame, text="", anchor="w", justify="left", font=custom_font)
        self.label_sent.grid(row=6, column=0, padx=DEF_MARGIN, pady=5, sticky=STICKY_LEFT)

        self.status_frame = tk.Frame(self.control_frame)
        self.status_frame.grid(row=10, column=0, padx=DEF_MARGIN, pady=0, sticky=STICKY_CENTER)
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_frame.grid_columnconfigure(1, weight=1)

        self.label_status = tk.Label(self.status_frame, text="Ready", font=custom_font, anchor="w", justify="left")
        self.label_status.grid(row=0, column=0, padx=0, pady=0, sticky=STICKY_LEFT)

        custom_font = font.Font(family="Consolas", size=12)
        self.label_seg = tk.Label(self.status_frame, text="0", font=custom_font)
        self.label_seg.grid(row=0, column=1, padx=0, pady=0, sticky=STICKY_RIGHT)

        self.update_stop()

    def on_click(self, e):
        if isinstance(e.widget, tk.Entry):
            return
        
        self.root.focus_set()

    def key_pressed(self, e):
        focus = self.root.focus_get()
        if focus is not None and focus != self.root:
            return
        
        if e.keysym == "x":
            self.command_stop_start()

        if e.keysym == "q":
            self.__del__()

    def comamnd_enter_pressed(self, e):
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
            custom_font = font.Font(family="Verdana Bold", size=18, weight="bold")
            self.stop_button = tk.Button(
                self.control_frame,
                text="START",
                font=custom_font,
                borderwidth=2,
                bg="white",
                fg="green" if connected else "gray",
                activebackground="white",
                activeforeground="green" if connected else "gray",
                relief="solid",
                command=self.command_stop_start,
            )
            self.stop_button.grid(row=9, column=0, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_CENTER)
            self.stop_button.config(padx=20, pady=5)

            if not connected:
                self.root.configure(bg="red")
            else:
                self.root.configure(bg="SystemButtonFace")
        else:
            self.update_status("Motor started.")
            custom_font = font.Font(family="Verdana Bold", size=18, weight="bold")
            self.stop_button = tk.Button(
                self.control_frame,
                text="STOP",
                font=custom_font,
                borderwidth=2,
                bg="yellow",
                fg="red",
                activebackground="yellow",
                activeforeground="red",
                relief="solid",
                command=self.command_stop_start,
            )
            self.stop_button.grid(row=9, column=0, padx=DEF_MARGIN, pady=DEF_MARGIN, sticky=STICKY_CENTER)
            self.stop_button.config(padx=20, pady=5)

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
        
        self.update_sent_command()
        self.update_received_command()
        self.update_wheel()

    def __del__(self):
        self.tracker.close()

    def update_received_command(self):
        if self.queue.has("r"):
            cmd = self.queue.get("r")
            if not self.received_text.get("1.0", tk.END).strip():
                self.received_text.insert(tk.END, "{}: {}".format(self.timeline_index, cmd))
            else:
                self.received_text.insert(tk.END, "\n{}: {}".format(self.timeline_index, cmd))
            
            self.label_received.config(text=cmd)
            self.timeline_index += 1

            if self.var_auto_scroll_received.get():
                self.received_text.see(tk.END)

    def update_sent_command(self):
        if self.queue.has("s"):
            cmd = self.queue.get("s")
            if not self.sent_text.get("1.0", tk.END).strip():
                self.sent_text.insert(tk.END, "{}: {}".format(self.timeline_index, cmd))
            else:
                self.sent_text.insert(tk.END, "\n{}: {}".format(self.timeline_index, cmd))
            
            self.label_sent.config(text=cmd)
            self.timeline_index += 1

            if self.var_auto_scroll_sent.get():
                self.sent_text.see(tk.END)
