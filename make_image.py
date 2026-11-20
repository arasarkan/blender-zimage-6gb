#!/usr/bin/env python3
"""Blender blockout or reference image -> high quality still, on a 6 GB GPU.

  python make_image.py scene.blend --prompt "..." [--camera Cam --frame 40]
  python make_image.py reference.jpg --prompt "..."          (follows its edges)

Z-Image-Turbo (6B) + Fun Union ControlNet draws a ~1 MP image that follows the
control (exact depth rendered from Blender, or the edges of a reference image),
then a 2K pass (RealESRGAN x4 -> half -> Z-Image at low denoise) adds detail.
Output: outputs/<name>/   (image_s<seed>_2k.png, compare.jpg, control.png, settings.json)
"""
import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comfy_client import CONFIG, Comfy, die, find_blender, log, saved_files  # noqa: E402

MODELS = {
    "unet": ("diffusion_models", "z_image_turbo_bf16.safetensors"),
    "clip": ("text_encoders", "qwen_3_4b.safetensors"),
    "vae": ("vae", "ae.safetensors"),
    "control": ("model_patches", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
    "upscale": ("upscale_models", "RealESRGAN_x4plus.safetensors"),
}
ASPECTS = {  # ~1 MP, multiples of 16
    "1:1": (1024, 1024), "3:2": (1216, 832), "2:3": (832, 1216), "16:9": (1344, 768),
    "9:16": (768, 1344), "4:5": (896, 1120), "5:4": (1120, 896), "21:9": (1536, 640),
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def fit_1mp(w, h):
    """Size with the reference's aspect ratio, ~1 MP, multiples of 16."""
    scale = (1024 * 1024 / (w * h)) ** 0.5
    return max(256, round(w * scale / 16) * 16), max(256, round(h * scale / 16) * 16)


def render_blend(a, out, width, height):
    blender = find_blender(a.blender)
    cmd = [blender, "-b", str(Path(a.source).resolve()), "--factory-startup", "-P",
           str(HERE / "blender" / "render_depth.py"), "--", "--out", str(out / "render"), "--frames", "1",
           "--width", str(width), "--height", str(height), "--depth-curve", a.depth_curve]
    for flag, value in (("camera", a.camera), ("start", a.frame), ("near", a.near), ("far", a.far)):
        if value is not None:
            cmd += [f"--{flag}", str(value)]
    log(f"rendering depth in Blender ({width}x{height})...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    done = [l for l in proc.stdout.splitlines() if l.startswith("DEPTH_DONE")]
    if proc.returncode or not done:
        print(proc.stdout[-3000:], proc.stderr[-3000:], file=sys.stderr)
        die("Blender render failed")
    info = json.loads(done[0].split(" ", 1)[1])
    log(f"depth range near={info['near']:.2f} far={info['far']:.2f} (override with --near/--far)")
    shutil.copy(out / "render" / "depth" / "0001.png", out / "control.png")
    if (out / "render" / "gray" / "0001.png").exists():
        shutil.copy(out / "render" / "gray" / "0001.png", out / "blockout.png")


def prepare_reference(a, out):
    from PIL import Image
    img = Image.open(a.source).convert("RGB")
    width, height = ASPECTS[a.aspect] if a.aspect else fit_1mp(*img.size)
    # center-crop to the target aspect, then resize
    src_ratio, dst_ratio = img.width / img.height, width / height
    if src_ratio > dst_ratio:
        new_w = round(img.height * dst_ratio)
        img = img.crop(((img.width - new_w) // 2, 0, (img.width - new_w) // 2 + new_w, img.height))
    elif src_ratio < dst_ratio:
        new_h = round(img.width / dst_ratio)
        img = img.crop((0, (img.height - new_h) // 2, img.width, (img.height - new_h) // 2 + new_h))
    img.resize((width, height), Image.LANCZOS).save(out / "reference.png")
    return width, height


def make_compare(out, result_name):
    """Input (blockout / reference) next to the result, for a quick look or sharing."""
    from PIL import Image
    left = next((out / f for f in ("blockout.png", "reference.png", "control.png") if (out / f).exists()), None)
    if not left:
        return
    right = Image.open(out / result_name).convert("RGB")
    h = 1024
    right = right.resize((round(right.width * h / right.height), h), Image.LANCZOS)
    left = Image.open(left).convert("RGB").resize(right.size, Image.LANCZOS)
    canvas = Image.new("RGB", (right.width * 2 + 8, h), (240, 240, 240))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (right.width + 8, 0))
    canvas.save(out / "compare.jpg", quality=92)


def build_workflow(a, control_file, width, height, seed, prefix):
    m = {k: v[1] for k, v in MODELS.items()}
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "lumina2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "patch": {"class_type": "ModelPatchLoader", "inputs": {"name": m["control"]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": control_file}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": a.prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save1k": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"{prefix}_1k"}},
    }
    control = ["img", 0]
    if a.control == "canny":
        wf["canny"] = {"class_type": "Canny", "inputs": {
            "image": ["img", 0], "low_threshold": a.canny_low, "high_threshold": a.canny_high}}
        control = ["canny", 0]
        wf["save_ctrl"] = {"class_type": "SaveImage", "inputs": {"images": control, "filename_prefix": f"{prefix}_control"}}
    wf["cn"] = {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
        "model": ["unet", 0], "model_patch": ["patch", 0], "vae": ["vae", 0], "image": control,
        "strength": a.strength, "end_percent": a.control_end}}
    wf["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["cn", 0], "shift": 3.0}}
    wf["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
        "seed": seed, "steps": a.steps, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple",
        "denoise": 1.0}}
    if not a.no_upscale:
        # 2K detail pass: RealESRGAN x4 -> x0.5 (= 2x) -> Z-Image img2img at low denoise
        wf.update({
            "up_model": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": m["upscale"]}},
            "up": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["up_model", 0], "image": ["decode", 0]}},
            "half": {"class_type": "ImageScaleBy", "inputs": {"image": ["up", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
            # tiled encode/decode: a 2K image in one piece overflows 6 GB during VAE decode
            "enc": {"class_type": "VAEEncodeTiled", "inputs": {
                "pixels": ["half", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                "temporal_size": 64, "temporal_overlap": 8}},
            "shift2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.0}},
            "refine": {"class_type": "KSampler", "inputs": {
                "model": ["shift2", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["enc", 0],
                "seed": seed, "steps": 5, "cfg": 1.0, "sampler_name": "dpmpp_2m_sde", "scheduler": "beta",
                "denoise": a.detail}},
            "decode2k": {"class_type": "VAEDecodeTiled", "inputs": {
                "samples": ["refine", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                "temporal_size": 64, "temporal_overlap": 8}},
            "save2k": {"class_type": "SaveImage", "inputs": {"images": ["decode2k", 0], "filename_prefix": f"{prefix}_2k"}},
        })
    return wf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?", help=".blend scene, or a reference image (jpg/png/webp); "
                   "default: the example blockout")
    p.add_argument("--prompt", help="what the image should look like (asked if missing)")
    p.add_argument("--aspect", choices=list(ASPECTS), help="output shape (default: 3:2 for .blend, the reference's own)")
    p.add_argument("--count", type=int, default=1, help="number of variations (seed, seed+1, ...)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--control", choices=["depth", "canny"], help="default: depth for .blend, canny for images")
    p.add_argument("--strength", type=float, default=0.8, help="how strictly to follow the control (0-1.5)")
    p.add_argument("--control-end", type=float, default=1.0, help="stop following the control after this fraction of steps")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--detail", type=float, default=0.33, help="denoise of the 2K detail pass (0.2 subtle - 0.5 strong)")
    p.add_argument("--no-upscale", action="store_true", help="only the ~1 MP image (about 4x faster)")
    p.add_argument("--control-only", action="store_true",
                   help="only prepare the control image (depth / reference), no ComfyUI -- a quick check")
    b = p.add_argument_group(".blend options")
    b.add_argument("--camera", help="camera object (default: scene camera)")
    b.add_argument("--frame", type=float, help="scene frame to render (default: scene start)")
    b.add_argument("--near", type=float)
    b.add_argument("--far", type=float)
    b.add_argument("--depth-curve", choices=["linear", "inverse"], default="linear",
                   help="linear works best for stills")
    b.add_argument("--blender")
    i = p.add_argument_group("reference image options")
    i.add_argument("--canny-low", type=float, default=0.15)
    i.add_argument("--canny-high", type=float, default=0.3)
    r = p.add_argument_group("run")
    r.add_argument("--name")
    r.add_argument("--out-dir", default=str(HERE / "outputs"))
    r.add_argument("--server", default=CONFIG.get("server") or "http://127.0.0.1:8188")
    a = p.parse_args()
    if not a.source:
        a.source = str(HERE / "examples" / "test_blockout_24fps.blend")
        print(f"No file given: using the example blockout ({a.source}).")
    if not Path(a.source).is_file():
        p.error(f"{a.source} not found")
    a.kind = "blend" if Path(a.source).suffix.lower() == ".blend" else "image"
    if a.kind == "image" and Path(a.source).suffix.lower() not in IMAGE_EXT:
        p.error("source must be a .blend or an image file")
    a.control = a.control or ("depth" if a.kind == "blend" else "canny")
    if a.kind == "image" and a.control == "depth":
        p.error("depth from a reference image needs a depth estimator; use --control canny")
    if not a.prompt and not a.control_only:
        if not sys.stdin.isatty():
            p.error("--prompt is required")
        a.prompt = input("Describe the image (English works best): ").strip()
    return a


def main():
    a = parse_args()
    name = a.name or dt.datetime.now().strftime("img_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[\w.-]+", name):
        die("--name may only contain letters, digits, '_', '-' and '.'")
    out = Path(a.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    if a.kind == "blend":
        width, height = ASPECTS[a.aspect or "3:2"]
        render_blend(a, out, width, height)
        control_src = out / "control.png"
    else:
        width, height = prepare_reference(a, out)
        control_src = out / "reference.png"
    if a.control_only:
        log(f"control image ready: {control_src}" + (" (canny is applied in ComfyUI)" if a.control == "canny" else ""))
        return
    comfy = Comfy(a.server)
    comfy.check(["QwenImageDiffsynthControlnet", "ModelPatchLoader", "ModelSamplingAuraFlow", "Canny",
                 "VAEDecodeTiled", "UpscaleModelLoader"], MODELS.values())
    log(f"{width}x{height}, control={a.control} strength={a.strength}, "
        f"{'2K detail pass' if not a.no_upscale else 'no upscale'}, {a.count} image(s)")
    control_file = comfy.upload(control_src, f"makeimg_{name}_{control_src.name}")

    results = []
    for k in range(a.count):
        seed = a.seed + k
        prefix = f"make_image/{name}_s{seed}"
        outputs, secs = comfy.run(build_workflow(a, control_file, width, height, seed, prefix))
        for node, label in (("save1k", "1k"), ("save2k", "2k"), ("save_ctrl", "control")):
            for item in saved_files(outputs, node):
                dest = out / ("control.png" if label == "control" else f"image_s{seed}_{label}.png")
                comfy.download(item, dest)
        final = out / f"image_s{seed}_{'1k' if a.no_upscale else '2k'}.png"
        results.append({"seed": seed, "file": final.name, "seconds": round(secs)})
        log(f"saved {final}")

    make_compare(out, results[0]["file"])
    settings = {k: v for k, v in vars(a).items() if k not in ("server",)}
    settings.update(width=width, height=height, results=results)
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    log(f"done: {out}")


if __name__ == "__main__":
    main()
