import sys
import os
import subprocess
import time
import threading
import tkinter as tk
import webbrowser
from PIL import Image, ImageTk, ImageDraw
import socket

def find_free_port(start=8501, end=8600):

    
    for port in range(start, end):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            try:
                s.bind(("127.0.0.1", port))
                return port

            except OSError:
                continue

    return start
    

    

class SplashScreen:


    WIDTH = 720
    HEIGHT = 480
    HOLD_TIME = 4

    def __init__(self):

        self.root = tk.Tk()

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        x = (sw - self.WIDTH) // 2
        y = (sh - self.HEIGHT) // 2

        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

        self._build_ui()

        self.launched = False



    def _create_gradient(self):

        img = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(img)

        for y in range(self.HEIGHT):

            t = y / self.HEIGHT

            r = int(8 + t * 8)
            g = int(12 + t * 14)
            b = int(20 + t * 28)

            draw.line([(0, y), (self.WIDTH, y)], fill=(r, g, b))

        return img



    def _build_ui(self):

        canvas = tk.Canvas(
            self.root,
            width=self.WIDTH,
            height=self.HEIGHT,
            bg="#0a0e17",
            highlightthickness=0,
        )

        canvas.pack(fill="both", expand=True)

        self.canvas = canvas

        bg_img = ImageTk.PhotoImage(self._create_gradient())

        self.bg = bg_img

        canvas.create_image(0, 0, anchor="nw", image=bg_img)

        canvas.create_text(
            self.WIDTH // 2,
            180,
            text="QuanSen",
            fill="#00d0ff",
            font=("Segoe UI", 44, "bold"),
        )

        canvas.create_text(
            self.WIDTH // 2,
            230,
            text="Quantitative Portfolio Optimizer",
            fill="#7fa7c6",
            font=("Segoe UI", 14),
        )

        self.bar_bg = canvas.create_rectangle(
            100,
            350,
            self.WIDTH - 100,
            360,
            fill="#111e2e",
            outline="",
        )

        self.bar = canvas.create_rectangle(
            100,
            350,
            100,
            360,
            fill="#19d3ff",
            outline="",
        )

        self.status = canvas.create_text(
            self.WIDTH // 2,
            380,
            text="Initializing...",
            fill="#8fbad4",
            font=("Segoe UI", 10),
        )


    # --------------------------------------------------
    # Fade in
    # --------------------------------------------------
    def _fade_in(self, step=0):

        self.root.attributes("-alpha", min(step / 20, 1))

        if step < 20:
            self.root.after(20, self._fade_in, step + 1)
        else:
            self._animate_bar(time.time())


    # --------------------------------------------------
    # Progress animation
    # --------------------------------------------------
    def _animate_bar(self, start_time):

        elapsed = time.time() - start_time

        ratio = min(elapsed / self.HOLD_TIME, 1)

        bar_width = (self.WIDTH - 200) * ratio

        self.canvas.coords(
            self.bar,
            100,
            350,
            100 + bar_width,
            360,
        )

        if ratio < 1:

            self.root.after(30, self._animate_bar, start_time)

        else:

            if not self.launched:

                self.launched = True

                threading.Thread(
                    target=self.launch_streamlit,
                    daemon=True,
                ).start()

            self.root.after(400, self.root.destroy)


    # --------------------------------------------------
    # Launch Streamlit
    # --------------------------------------------------
    def launch_streamlit(self):

        try:

            port = find_free_port()

            base_dir = os.path.dirname(os.path.abspath(__file__))

            gui_path = os.path.join(
                base_dir,
                "versions",
                "v2.0",
                "main.py",
            )

            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    gui_path,
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "false",
                ]
            )

            time.sleep(2)

            

        except Exception as e:

            print("Launcher error:", e)


    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self):

        self.root.after(50, self._fade_in)

        self.root.mainloop()
    

    # --------------------------------------------------

    # Main

    # --------------------------------------------------

def main():

    
    splash = SplashScreen()

    splash.run()
        

if __name__ == "__main__":
    main()
    
