"""The PC side. Started at Windows login by local/install.ps1 (runs hidden, low priority).

Trading itself runs 24/7 in the cloud (GitHub Actions). While this PC is on, this loop
adds the stronger local brain: every hour it pulls the bot's latest state from GitHub,
lets the local LLM review it, asks CLM for a second opinion every few hours, and pushes
the results back. It runs from its own clone of the repo (<project>/.runner), so git
operations here never touch the project folder, and it restarts itself when the code
on GitHub changes.
"""

import ctypes
import os
import socket
import subprocess
import sys
import time
import traceback

import requests

from . import brain, clm, config

LOG = config.CACHE_DIR / "local.log"
BRAIN_EVERY_S = 3600
CLM_EVERY_S = 4 * 3600
PUSH = ["data/brain.json", "data/brain_journal.jsonl", "data/clm.json", "data/clm_log.jsonl"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_lock = socket.socket()  # holding this port = "an instance is running"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=300,
                       creationflags=NO_WINDOW)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r


def head(path=""):
    return git("rev-parse", f"HEAD:{path}" if path else "HEAD").stdout.strip()


def pull():
    """Mirror GitHub exactly (this clone is only used by this loop)."""
    git("fetch", "-q", "origin")
    git("reset", "--hard", "-q", "origin/main")


def push(msg):
    git("add", *[p for p in PUSH if (config.ROOT / p).exists()])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    git("commit", "-q", "-m", msg)
    for attempt in range(4):
        if git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            log("pushed " + msg)
            return
        if git("pull", "--rebase", "-q", "origin", "main", check=False).returncode != 0:
            git("rebase", "--abort", check=False)
            log("conflict with a cloud update; this round's notes are dropped, next hour will redo them")
            pull()
            return
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("push failed 4 times")


def ollama_ready():
    try:
        return requests.get(config.OLLAMA_URL + "/api/tags", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def restart_if_code_changed(code_before):
    if head("bot") != code_before:
        log("new code on GitHub; restarting")
        _lock.close()  # free the single-instance port before the new copy starts
        subprocess.Popen([sys.executable, "-m", "bot", "local"], cwd=config.ROOT, creationflags=NO_WINDOW)
        sys.exit(0)


def keep_cloud_schedule_alive():
    # GitHub pauses scheduled workflows in repos without activity for 60 days; re-enable just in case.
    for wf in ("bot.yml", "brain.yml"):
        subprocess.run(["gh", "workflow", "enable", wf], cwd=config.ROOT, capture_output=True, creationflags=NO_WINDOW)


def main():
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if LOG.exists() and LOG.stat().st_size > 5_000_000:
        LOG.write_text("", encoding="utf-8")
    sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)
    try:
        _lock.bind(("127.0.0.1", 47291))  # single instance
    except OSError:
        log("already running; exiting")
        return
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # below normal
    log(f"started (brain model {config.LOCAL_BRAIN_MODEL}, CLM {'on' if config.CLM_ENABLED else 'off'})")
    while True:  # wait for network after boot
        try:
            pull()
            break
        except Exception as exc:
            log(f"waiting for network/GitHub: {exc}")
            time.sleep(60)
    code = head("bot")
    keep_cloud_schedule_alive()
    last_brain = last_clm = 0.0
    while True:
        now = time.time()
        if now - last_brain >= BRAIN_EVERY_S:
            last_brain = now
            try:
                pull()
                restart_if_code_changed(code)
                did = []
                for _ in range(30):  # Ollama may still be starting right after login
                    if ollama_ready():
                        break
                    time.sleep(10)
                if ollama_ready():
                    brain.run(config.LOCAL_BRAIN_MODEL, "local")
                    did.append("brain")
                else:
                    log("Ollama is not running; skipping the brain this hour")
                if config.CLM_ENABLED and clm.installed() and now - last_clm >= CLM_EVERY_S:
                    last_clm = now
                    clm.run_once()
                    did.append("clm")
                if did:
                    push(f"local {'+'.join(did)} {time.strftime('%Y-%m-%d %H:%M')}")
            except SystemExit:
                raise
            except Exception:
                log("cycle failed:\n" + traceback.format_exc())
        time.sleep(60)
