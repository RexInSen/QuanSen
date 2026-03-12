"""
QuanSen — Splash Screen Launcher (Streamlit Edition)
=====================================================
Architecture:
  QuanSen.exe  (this file, windowed, onefile)
      └─ spawns ─► quansen_app\quansen_app.exe  (onedir Streamlit bundle)

Splash-loop is impossible: this file has zero Streamlit/portfolio code.
The app folder has zero splash code.
"""

import sys
import os
import subprocess
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw


def resource_path(rel):
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def get_app_cmd():
    """
    Frozen  →  quansen_app\quansen_app.exe   (onedir folder next to QuanSen.exe)
    Dev     →  python -m streamlit run gui_portfolio.py
    """
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        app_exe = os.path.join(exe_dir, "quansen_app", "quansen_app.exe")
        return [app_exe]
    else:
        gui_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_portfolio.py")
        return [
            sys.executable, "-m", "streamlit", "run", gui_py,
            "--global.developmentMode=false",
            "--server.port=8501",
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
        ]


class SplashScreen:
    W            = 600
    H            = 400
    HOLD_SECONDS = 3.2

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.root.configure(bg="#0a0e17")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}+{(sw-self.W)//2}+{(sh-self.H)//2}")

        ico = resource_path("quansen.ico")
        if os.path.exists(ico):
            try:
                self.root.iconbitmap(ico)
            except Exception:
                pass

        self._launched = False
        self._build_ui()

    def _make_gradient(self):
        img  = Image.new("RGB", (self.W, self.H))
        draw = ImageDraw.Draw(img)
        for y in range(self.H):
            t = y / self.H
            draw.line([(0, y), (self.W, y)],
                      fill=(int(10+t*6), int(14+t*10), int(23+t*14)))
        return img

    def _build_ui(self):
        c = tk.Canvas(self.root, width=self.W, height=self.H,
                      bg="#0a0e17", highlightthickness=0)
        c.pack(fill="both", expand=True)
        self.canvas = c

        self._bg_photo = ImageTk.PhotoImage(self._make_gradient())
        c.create_image(0, 0, anchor="nw", image=self._bg_photo)

        self._logo_photo = None
        logo_path = resource_path("quansen.png")
        if os.path.exists(logo_path):
            try:
                logo = (Image.open(logo_path).convert("RGBA")
                             .resize((155, 155), Image.LANCZOS))
                self._logo_photo = ImageTk.PhotoImage(logo)
                c.create_image(148, 168, anchor="center", image=self._logo_photo)
            except Exception:
                pass

        c.create_text(355, 115, text="QuanSen",
                      fill="#00c8ff", font=("Segoe UI", 40, "bold"), anchor="center")
        c.create_text(355, 162, text="Quantitative Portfolio Optimizer",
                      fill="#b0c8dc", font=("Segoe UI", 13), anchor="center")
        c.create_text(355, 190,
                      text="MPT  ·  Efficient Frontier  ·  CML  ·  Sharpe",
                      fill="#3d5a70", font=("Segoe UI", 9), anchor="center")
        c.create_text(355, 216, text="v1.0  —  by Amatra Sen",
                      fill="#2d4a5e", font=("Segoe UI", 9, "italic"), anchor="center")

        c.create_rectangle(285, 236, 425, 256,
                           fill="#0d1e30", outline="#1a3a58", width=1)
        c.create_text(355, 246, text="● STREAMLIT  POWERED",
                      fill="#00c8ff", font=("Segoe UI", 8), anchor="center")

        bx0, bx1, by, bh = 55, self.W - 55, 318, 5
        c.create_rectangle(bx0, by, bx1, by + bh,
                           fill="#111e2e", outline="#1a2d40", width=1)
        self._bar = c.create_rectangle(bx0, by, bx0, by + bh,
                                       fill="#00c8ff", outline="")
        self._bx0, self._bw = bx0, bx1 - bx0
        self._by,  self._bh = by,  bh

        self._status_id = c.create_text(self.W // 2, 340,
                                        text="Initializing...",
                                        fill="#2d4a60",
                                        font=("Segoe UI", 9),
                                        anchor="center")
        c.create_rectangle(1, 1, self.W - 2, self.H - 2,
                           outline="#0d2a44", width=1)

    _MESSAGES = [
        (0,  "Initializing..."),
        (25, "Loading optimizer engine..."),
        (50, "Preparing Streamlit server..."),
        (75, "Opening browser..."),
        (95, "Ready."),
    ]

    def _fade_in(self, step=0):
        self.root.attributes("-alpha", min(step / 18, 1.0))
        if step < 18:
            self.root.after(22, self._fade_in, step + 1)
        else:
            self._animate_bar(time.time())

    def _animate_bar(self, t0):
        elapsed = time.time() - t0
        ratio   = min(elapsed / self.HOLD_SECONDS, 1.0)
        pct     = int(ratio * 100)

        self.canvas.coords(self._bar,
                           self._bx0, self._by,
                           self._bx0 + int(self._bw * ratio), self._by + self._bh)

        msg = self._MESSAGES[0][1]
        for threshold, text in self._MESSAGES:
            if pct >= threshold:
                msg = text
        self.canvas.itemconfig(self._status_id, text=msg)

        if ratio < 1.0:
            self.root.after(28, self._animate_bar, t0)
        else:
            if not self._launched:
                self._launched = True
                threading.Thread(target=self._launch_app, daemon=True).start()
            self.root.after(350, self._fade_out, 18)

    def _launch_app(self):
        cmd = get_app_cmd()
        try:
            CREATE_NEW_CONSOLE = 0x00000010
            subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE, close_fds=True)
        except Exception as err:
            self.canvas.itemconfig(self._status_id,
                                   text=f"Launch failed: {err}",
                                   fill="#ff5252")
            time.sleep(4)

    def _fade_out(self, step):
        self.root.attributes("-alpha", max(step / 18, 0.0))
        if step > 0:
            self.root.after(22, self._fade_out, step - 1)
        else:
            self.root.after(40, self.root.destroy)

    def run(self):
        self.root.after(50, self._fade_in)
        self.root.mainloop()


def main():
    SplashScreen().run()


if __name__ == "__main__":
    main()
