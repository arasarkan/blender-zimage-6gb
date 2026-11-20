#!/usr/bin/env python3
"""One-time setup for blender-zimage-6gb.

Run it while ComfyUI is open (on Windows just double-click setup.bat):
  - finds your ComfyUI (its Python, its models folder) and Blender
  - downloads the 4 model files that are missing (official Comfy-Org repos,
    .safetensors only, resumable, SHA256-verified)
  - writes config.json so make_image.py / make_image.bat need no paths
  - renders a test depth image to check Blender works

Options: --yes (no questions), --server URL, --models-dir DIR, --blender PATH
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402  (Blender lookup, ComfyUI client)

CONFIG = HERE / "config.json"
# (key, models subfolder, Hugging Face repo, path in repo) -- official Comfy-Org repackages, plus the
# ControlNet from alibaba-pai (Alibaba, the makers of Z-Image)
MODELS = [
    ("unet", "diffusion_models", "Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("clip", "text_encoders", "Comfy-Org/z_image_turbo",
     "split_files/text_encoders/qwen_3_4b.safetensors"),
    ("vae", "vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    ("control", "model_patches", "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union",
     "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
    ("upscale", "upscale_models", "Comfy-Org/Real-ESRGAN_repackaged", "RealESRGAN_x4plus.safetensors"),
]
MIN_VRAM_GB = 5.5
USER_AGENT = "blender-zimage-6gb-setup"


def say(msg=""):
    print(msg, flush=True)


def ask(question, default=""):
    try:
        answer = input(f"{question} " + (f"[{default}] " if default else "")).strip()
    except EOFError:
        answer = ""
    return answer.strip('"') or default


def yes(question, assume_yes):
    return assume_yes or ask(f"{question} (Y/n)", "y").lower().startswith("y")


def fail(msg):
    say(f"\nERROR: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- ComfyUI discovery

def comfy_process_cmdline(port):
    """Command line of the process listening on the ComfyUI port (Windows only)."""
    if os.name != "nt":
        return None
    ps = (f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
          "Select-Object -First 1; if ($c) { $p = Get-CimInstance Win32_Process -Filter "
          "\"ProcessId=$($c.OwningProcess)\"; $p.ExecutablePath; $p.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out[0], out[1]) if len(out) >= 2 else None


def arg_value(argv, name):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else None
    for a in argv:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def models_dir_from_yaml(path):
    """base_path of the is_default entry in an extra_model_paths yaml."""
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    entries = [v for v in data.values() if isinstance(v, dict) and v.get("base_path")]
    entries.sort(key=lambda v: not v.get("is_default"))
    return Path(os.path.expandvars(entries[0]["base_path"])) if entries else None


def find_comfy_install(port):
    """-> (comfy_python, comfy_dir, models_dir); any may be None."""
    found = comfy_process_cmdline(port)
    if not found:
        return None, None, None
    exe, cmdline = found
    argv = shlex.split(cmdline, posix=False)
    argv = [a.strip('"') for a in argv]
    python = Path(argv[0]) if Path(argv[0]).is_absolute() and Path(argv[0]).is_file() else Path(exe)

    comfy_dir = None
    main = next((a for a in argv if a.replace("\\", "/").endswith("main.py")), None)
    candidates = []
    if main and Path(main).is_absolute():
        candidates.append(Path(main).parent)
    for parent in list(python.parents)[:4]:
        candidates += [parent, parent / "ComfyUI"]
        if main:
            candidates.append((parent / main).parent)
    for c in candidates:
        if (c / "main.py").is_file() and (c / "comfy").is_dir():
            comfy_dir = c
            break

    models_dir = None
    yaml_path = arg_value(argv, "--extra-model-paths-config")
    if yaml_path:
        models_dir = models_dir_from_yaml(yaml_path)
    if not models_dir and arg_value(argv, "--base-directory"):
        models_dir = Path(arg_value(argv, "--base-directory")) / "models"
    if not models_dir and comfy_dir:
        if (comfy_dir / "extra_model_paths.yaml").is_file():
            models_dir = models_dir_from_yaml(comfy_dir / "extra_model_paths.yaml")
        models_dir = models_dir or comfy_dir / "models"
    return python, comfy_dir, models_dir


def python_has_deps(python):
    try:
        r = subprocess.run([str(python), "-c", "import av, numpy, PIL"], capture_output=True, timeout=60)
        return r.returncode == 0
    except OSError:
        return False


# ---------------------------------------------------------------- downloads

def hf_get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def remote_info(repo, path):
    folder = path.rsplit("/", 1)[0] if "/" in path else ""
    with hf_get(f"https://huggingface.co/api/models/{repo}/tree/main/{folder}") as r:
        for entry in json.load(r):
            if entry["path"] == path:
                return entry["size"], entry.get("lfs", {}).get("oid")
    fail(f"{path} not found in {repo}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def download(repo, path, dest, size, digest):
    part = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    for attempt in range(1, 6):
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        try:
            with hf_get(url, {"Range": f"bytes={done}-"}, timeout=120) as r, open(part, "ab") as f:
                if done and r.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                    done = 0
                start, start_bytes, last = time.time(), done, 0
                while chunk := r.read(4 * 2**20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 1:
                        last = time.time()
                        speed = (done - start_bytes) / max(0.001, last - start) / 1e6
                        print(f"\r    {done / 1e9:6.2f} / {size / 1e9:.2f} GB  {speed:5.1f} MB/s",
                              end="", flush=True)
            print(f"\r    {done / 1e9:6.2f} / {size / 1e9:.2f} GB", flush=True)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            say(f"\n    connection problem ({e}); retrying {attempt}/5 ...")
            time.sleep(3 * attempt)
    if not part.exists() or part.stat().st_size != size:
        fail(f"download of {dest.name} incomplete; run setup again to resume.")
    if digest:
        say("    verifying SHA256 ...")
        if sha256(part) != digest:
            part.unlink()
            fail(f"{dest.name} failed the checksum; deleted it, run setup again.")
    part.replace(dest)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188"))
    p.add_argument("--models-dir", help="ComfyUI models folder (default: detected)")
    p.add_argument("--blender", help="path to blender executable (default: detected)")
    p.add_argument("--yes", action="store_true", help="answer yes to all questions")
    a = p.parse_args()

    say("=== blender-zimage-6gb setup ===\n")

    # 1. ComfyUI
    say("[1/5] ComfyUI")
    comfy = comfy_client.Comfy(a.server)
    try:
        with urllib.request.urlopen(a.server.rstrip("/") + "/system_stats", timeout=10) as r:
            stats = json.load(r)
    except (urllib.error.URLError, OSError):
        fail(f"ComfyUI is not running at {a.server}. Start ComfyUI, wait until it has loaded, "
             "then run setup again.")
    dev = (stats.get("devices") or [{}])[0]
    gpu = re.sub(r"^\w+:\d+\s+|\s*:\s*\w+$", "", dev.get("name", "?"))  # "cuda:0 NAME : allocator" -> NAME
    vram = dev.get("vram_total", 0) / 2**30
    say(f"    ComfyUI {stats['system'].get('comfyui_version', '?')}, GPU: {gpu}, "
        f"{vram:.1f} GB VRAM")
    if dev.get("type") != "cuda":
        say("    WARNING: no NVIDIA GPU in use by ComfyUI; this project is built for NVIDIA cards.")
    elif vram < MIN_VRAM_GB:
        say(f"    WARNING: less than {MIN_VRAM_GB} GB VRAM; generation may run out of memory.")

    info = comfy.request("/object_info")
    needed = ["QwenImageDiffsynthControlnet", "ModelPatchLoader", "ModelSamplingAuraFlow", "Canny",
              "VAEDecodeTiled", "VAEEncodeTiled", "UpscaleModelLoader", "ImageUpscaleWithModel"]
    missing_nodes = [n for n in needed if n not in info]
    if missing_nodes:
        fail(f"your ComfyUI is too old (missing {', '.join(missing_nodes)}). Update ComfyUI and run setup again.")

    port = urllib.parse.urlparse(a.server).port or 80
    comfy_python, comfy_dir, models_dir = find_comfy_install(port)
    if a.models_dir:
        models_dir = Path(a.models_dir)
    if comfy_dir:
        say(f"    install: {comfy_dir}")
    python = comfy_python if comfy_python and python_has_deps(comfy_python) else None
    if not python and python_has_deps(sys.executable):
        python = Path(sys.executable)
    if not python:
        fail("could not find a Python with PyAV/numpy/Pillow. Run this script with ComfyUI's Python, "
             "or install them: pip install -r requirements.txt")
    say(f"    python: {python}")

    # 2. models folder
    say("\n[2/5] Models folder")
    while not models_dir or not Path(models_dir).is_dir():
        if models_dir:
            say(f"    {models_dir} does not exist.")
        if a.yes:
            fail("could not detect the ComfyUI models folder; pass --models-dir.")
        models_dir = Path(ask("    Paste the path of your ComfyUI 'models' folder:"))
    models_dir = Path(models_dir).resolve()
    say(f"    {models_dir}")

    # 3. models
    say("\n[3/5] Models")
    todo, total = [], 0
    for key, folder, repo, path in MODELS:
        name = path.rsplit("/", 1)[-1]
        dest = models_dir / folder / name
        if dest.is_file() or name in comfy.request(f"/models/{folder}"):
            say(f"    ok       {folder}/{name}")
            continue
        size, digest = remote_info(repo, path)
        todo.append((repo, path, dest, size, digest))
        total += size
        say(f"    missing  {folder}/{name} ({size / 1e9:.2f} GB)")
    if todo:
        free = shutil.disk_usage(models_dir).free
        say(f"\n    To download: {total / 1e9:.1f} GB from huggingface.co (Comfy-Org). Free disk: {free / 1e9:.0f} GB")
        if free < total * 1.05:
            fail("not enough free disk space.")
        if not yes("    Download now?", a.yes):
            fail("models are required; run setup again when ready.")
        for i, (repo, path, dest, size, digest) in enumerate(todo, 1):
            say(f"\n    ({i}/{len(todo)}) {dest.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            download(repo, path, dest, size, digest)
            say("    done")

    # 4. Blender
    say("\n[4/5] Blender")
    blender = comfy_client.find_blender(a.blender, required=False)
    while not blender:
        if a.yes:
            fail("Blender not found; install it or pass --blender PATH.")
        entered = ask("    Blender not found. Paste the path of blender.exe (or install Blender and press Enter):")
        blender = comfy_client.find_blender(entered or None, required=False)
    version = subprocess.run([blender, "-b", "--factory-startup", "--version"], capture_output=True,
                             text=True, errors="replace").stdout
    m = re.search(r"Blender (\d+)\.(\d+)", version)
    say(f"    {blender} ({m.group(0) if m else 'unknown version'})")
    if m and (int(m.group(1)), int(m.group(2))) < (4, 2):
        say("    WARNING: Blender 4.2 or newer is recommended.")

    CONFIG.write_text(json.dumps({
        "server": a.server, "python": str(python), "blender": blender, "models_dir": str(models_dir),
        "comfyui_dir": str(comfy_dir) if comfy_dir else None,
        "gpu": gpu, "vram_gb": round(vram, 1),
    }, indent=2))

    # 5. check
    say("\n[5/5] Test render (depth image from the example blockout, a few seconds)")
    r = subprocess.run([str(python), str(HERE / "make_image.py"), "--control-only", "--name", "setup_check"])
    if r.returncode:
        fail("test render failed; see the messages above.")

    say("\n=== Setup complete ===")
    say("Make an image:")
    if os.name == "nt":
        say("  - drag a .blend file or a reference image onto make_image.bat")
        say("  - or double-click make_image.bat to try the example blockout")
    say(f'  - or: "{python}" make_image.py scene.blend --prompt "..."')


if __name__ == "__main__":
    main()
