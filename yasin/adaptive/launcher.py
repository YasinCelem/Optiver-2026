"""
Start/stop the bot from a Jupyter notebook WITHOUT running it inside the kernel.

Running a bot directly in a notebook hangs it (too much output). This starts the
bot as a separate background process that writes to a log file; the notebook
only reads that file. The bot keeps running if you close the notebook.

In a notebook cell:

    import sys; sys.path.insert(0, "yasin/adaptive")   # path from your repo root
    import launcher

    launcher.start()      # start the bot in the background
    launcher.tail()       # show the last lines of its output (run again to refresh)
    launcher.status()     # running or not
    launcher.stop()       # clean stop: pulls orders, saves what it learned
"""
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PID_FILE = os.path.join(DATA_DIR, "bot.pid")
LOG_FILE = os.path.join(DATA_DIR, "bot.log")


_PROC = None          # Popen handle when started from this kernel


def _pid():
    if _PROC is not None and _PROC.poll() is not None:     # exited (reaps the zombie too)
        return None
    try:
        with open(PID_FILE) as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)                 # signal 0 = "does it exist?"
    except OSError:
        return None
    return pid


def status():
    pid = _pid()
    print(f"bot running (pid {pid})" if pid else "bot not running")
    return pid is not None


def start():
    """Start the bot. Refuses if one is already running (two would kick each other off)."""
    global _PROC
    if _pid():
        print("Already running; use launcher.stop() first.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    log = open(LOG_FILE, "a")
    log.write(f"\n\n######## start {time.strftime('%Y-%m-%d %H:%M:%S')} ########\n")
    log.flush()
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "bot.py")],
                            stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                            start_new_session=True)
    _PROC = proc
    with open(PID_FILE, "w") as fh:
        fh.write(str(proc.pid))
    time.sleep(3)
    if proc.poll() is not None:
        print(f"Bot exited straight away (code {proc.returncode}). Output:")
        tail(30)
    else:
        print(f"Bot started (pid {proc.pid}). Log: {LOG_FILE}")


def tail(n=40):
    """Print the last n lines of the bot's output."""
    try:
        with open(LOG_FILE, errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        print("No log yet.")
        return
    print("".join(lines[-n:]))


def stop(timeout=15):
    """Ctrl+C equivalent: the bot pulls its orders and saves brain.json before exiting."""
    pid = _pid()
    if not pid:
        print("Not running.")
        return
    os.kill(pid, signal.SIGINT)
    for _ in range(int(timeout * 10)):
        time.sleep(0.1)
        if not _pid():
            print("Stopped cleanly.")
            tail(15)
            return
    os.kill(pid, signal.SIGTERM)
    print("Did not stop in time; terminated (open orders are pulled by the exchange on disconnect).")
