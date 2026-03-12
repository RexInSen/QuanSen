"""
quansen_app.py — Streamlit bootstrap for PyInstaller
"""

import sys
import os
import socket


def resource(rel):
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def find_free_port(start=8501, end=8600):
    """Find a free port in range, starting from start."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    return start  # fallback


def main():
    # ── Must be set BEFORE importing streamlit ────────────────
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_DEVELOPMENT_MODE"] = "false"

    # ── Find a free port automatically ───────────────────────
    port = find_free_port(8501, 8600)
    print(f"Starting QuanSen on port {port}...")

    # ── Streamlit static assets ───────────────────────────────
    static_dir = resource(os.path.join("streamlit", "static"))
    if os.path.isdir(static_dir):
        os.environ["STREAMLIT_STATIC_FOLDER"] = static_dir

    # ── Streamlit env config ──────────────────────────────────
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"]             = "false"
    os.environ["STREAMLIT_SERVER_PORT"]                 = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"]              = "localhost"

    # ── Write config.toml fallback ────────────────────────────
    config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")
    with open(config_path, "w") as f:
        f.write(f"""
[global]
developmentMode = false

[server]
port = {port}
address = "localhost"
headless = false

[browser]
gatherUsageStats = false
""")

    # ── GUI script path ───────────────────────────────────────
    gui_script = resource("gui_portfolio.py")

    # ── Launch Streamlit in-process ───────────────────────────
    sys.argv = [
        "streamlit", "run", gui_script,
        "--global.developmentMode=false",
        f"--server.port={port}",
        "--server.address=localhost",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]

    from streamlit.web import cli as st_cli
    st_cli.main()


if __name__ == "__main__":
    main()
