"""Shared helpers: logging, config.json, Blender lookup and a small ComfyUI API client."""
import datetime as dt
import glob
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_config():
    """config.json written by setup.py (paths found on this machine)."""
    try:
        return json.loads((HERE / "config.json").read_text())
    except (OSError, ValueError):
        return {}


CONFIG = load_config()


def find_blender(explicit, required=True):
    candidates = [explicit, CONFIG.get("blender"), os.environ.get("BLENDER"), shutil.which("blender")]
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
        r"C:\Program Files\Steam\steamapps\common\Blender\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/.steam/steam/steamapps/common/Blender/blender"),
        "/usr/bin/blender", "/snap/bin/blender",
    ]
    for pat in patterns:
        candidates += sorted(glob.glob(pat), reverse=True)  # newest version first
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    if required:
        die("Blender not found. Run setup again, pass --blender PATH or set the BLENDER environment variable.")
    return None


def saved_files(outputs, node_id):
    items = []
    for value in outputs.get(node_id, {}).values():
        if isinstance(value, list):
            items += [i for i in value if isinstance(i, dict) and "filename" in i]
    return items


class Comfy:
    def __init__(self, server):
        self.server = server.rstrip("/")

    def request(self, path, data=None, headers=None, raw=False, timeout=60):
        req = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            die(f"ComfyUI {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:3000]}")
        except urllib.error.URLError as e:
            die(f"cannot reach ComfyUI at {self.server} ({e.reason}). Is it running?")
        return body if raw else json.loads(body)

    def post_json(self, path, payload):
        return self.request(path, json.dumps(payload).encode(), {"Content-Type": "application/json"})

    def upload(self, file_path, name, content_type="image/png"):
        boundary = uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
             f"Content-Type: {content_type}\r\n\r\n").encode(),
            Path(file_path).read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        res = self.request("/upload/image", b"".join(parts),
                           {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def check(self, nodes, models):
        """Fail early if core nodes or model files are missing. models: [(folder, filename)]."""
        stats = self.request("/system_stats")
        dev = (stats.get("devices") or [{}])[0]
        log(f"ComfyUI {stats['system'].get('comfyui_version', '?')} on {dev.get('name', '?')}")
        info = self.request("/object_info")
        missing = [n for n in nodes if n not in info]
        if missing:
            die(f"your ComfyUI is missing core nodes {missing}; update ComfyUI.")
        for folder, filename in models:
            if filename not in self.request(f"/models/{folder}"):
                die(f"model {filename} not found in models/{folder}/. Run setup.bat (or setup.py) again.")
        free = dev.get("vram_free", 0) / 2**20
        if free and free < 4500:
            log(f"WARNING: only {free:.0f} MB VRAM free. Close other GPU apps if it runs out of memory.")

    def run(self, workflow):
        prompt_id = self.post_json("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})["prompt_id"]
        log(f"queued {prompt_id}")
        start, last = time.time(), time.time()
        try:
            while True:
                hist = self.request(f"/history/{prompt_id}").get(prompt_id)
                if hist and hist.get("status", {}).get("completed") is not None:
                    break
                if time.time() - last > 60:
                    last = time.time()
                    log(f"  generating... {(last - start) / 60:.0f} min")
                time.sleep(2)
        except KeyboardInterrupt:
            self.request("/interrupt", b"")
            die("interrupted; ComfyUI job cancelled.")
        status = hist["status"]
        if status.get("status_str") != "success":
            for kind, msg in status.get("messages", []):
                if kind == "execution_error":
                    print(json.dumps(msg, indent=2)[:4000], file=sys.stderr)
            die(f"ComfyUI job failed ({status.get('status_str')}). If ComfyUI crashed, restart it and try "
                "--no-upscale or a smaller --aspect.")
        log(f"done in {time.time() - start:.0f} s")
        return hist["outputs"], time.time() - start

    def download(self, item, dest):
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        Path(dest).write_bytes(self.request(f"/view?{q}", raw=True, timeout=300))
