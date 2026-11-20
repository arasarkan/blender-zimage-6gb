"""Render a depth control sequence (and a gray preview) from a Blender scene.

Run headless (make_video.py does this for you):

  blender -b [scene.blend] -P blender/render_depth.py -- --out DIR --frames 81 [options]

Without a .blend (or with --test-scene) a built-in gray blockout with a
camera dolly is used.

Frames are sampled at 16 fps (Wan 2.1's native rate) from the scene's own
timeline, whatever fps the scene was animated at: a scene animated at 24 fps
is sampled with sub-frame precision, so timing is preserved.

Output:
  DIR/depth/0001.png ...   near = white, far = black (Depth Anything convention)
  DIR/gray/0001.png ...    Workbench gray preview (unless --no-preview)
  DIR/depth_info.json      near/far used, frame mapping
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

WAN_FPS = 16


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="render_depth.py")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, required=True, help="number of 16 fps frames (4n+1)")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--start", type=float, default=None, help="scene frame to start at (default: scene start)")
    p.add_argument("--camera", default=None, help="camera object name (default: scene camera)")
    p.add_argument("--near", type=float, default=None, help="depth mapped to white (default: auto)")
    p.add_argument("--far", type=float, default=None, help="depth mapped to black (default: auto)")
    p.add_argument("--depth-curve", choices=["linear", "inverse"], default="linear",
                   help="inverse = disparity like Depth Anything; better for close-up / interior shots")
    p.add_argument("--test-scene", action="store_true", help="build the built-in test blockout")
    p.add_argument("--no-preview", action="store_true", help="skip the gray preview pass")
    return p.parse_args(argv)


def build_test_scene(frames):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = WAN_FPS
    scene.frame_start, scene.frame_end = 1, frames

    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
    for x, y, sx, sy, sz in [
        (-4, 4, 1.5, 1.5, 3), (3, 7, 2, 1, 1.5), (-1, 12, 3, 3, 5),
        (6, 14, 1, 1, 4), (-7, 16, 2, 4, 2), (1, 20, 6, 1, 3),
    ]:
        bpy.ops.mesh.primitive_cube_add(location=(x, y, sz))
        bpy.context.object.scale = (sx, sy, sz)
    for x, y in [(-2, 2), (2, 3.5), (0, 9)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=4, location=(x, y, 2))

    target = bpy.data.objects.new("CamTarget", None)
    scene.collection.objects.link(target)
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.lens = 28
    scene.collection.objects.link(cam)
    scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    for frame, cam_loc, tgt_loc in [
        (1, (0, -10, 2.5), (0, 10, 1.5)),
        (frames, (3, -2, 2.0), (1, 14, 1.5)),
    ]:
        cam.location, target.location = cam_loc, tgt_loc
        cam.keyframe_insert("location", frame=frame)
        target.keyframe_insert("location", frame=frame)


def scene_times(scene, frames, start):
    """Scene frame (float) for each 16 fps output frame."""
    scene_fps = scene.render.fps / scene.render.fps_base
    step = scene_fps / WAN_FPS
    start = scene.frame_start if start is None else start
    return [start + i * step for i in range(frames)]


def set_time(scene, t):
    whole = math.floor(t)
    scene.frame_set(whole, subframe=t - whole)


def auto_near_far(scene, cam, times):
    """Camera-space depth range of visible mesh bounding boxes over the shot."""
    depths = []
    for t in times[:: max(1, len(times) // 8)] + [times[-1]]:
        set_time(scene, t)
        cam_inv = cam.matrix_world.inverted()
        for obj in scene.objects:
            if obj.type != "MESH" or obj.hide_render:
                continue
            for corner in obj.bound_box:
                z = -(cam_inv @ (obj.matrix_world @ Vector(corner))).z
                if z > 0:
                    depths.append(z)
    clip_end = cam.data.clip_end
    if not depths:
        return 0.5, min(50.0, clip_end)
    depths.sort()
    near = max(cam.data.clip_start, depths[int(len(depths) * 0.02)])
    far = min(clip_end, depths[int(len(depths) * 0.90)])
    if far <= near * 1.5:
        far = near * 4
    return near, far


def depth_material(near, far, curve="linear"):
    mat = bpy.data.materials.new("WanDepthOverride")
    if mat.node_tree is None:  # Blender < 5 creates the node tree lazily
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    camdata = nt.nodes.new("ShaderNodeCameraData")
    remap = nt.nodes.new("ShaderNodeMapRange")
    remap.clamp = True
    depth_socket = camdata.outputs["View Z Depth"]
    if curve == "inverse":
        # disparity (1/z), like Depth Anything: much more contrast close to the camera
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "DIVIDE"
        inv.inputs[0].default_value = 1.0
        nt.links.new(depth_socket, inv.inputs[1])
        depth_socket = inv.outputs["Value"]
        remap.inputs["From Min"].default_value = 1.0 / far
        remap.inputs["From Max"].default_value = 1.0 / near
        remap.inputs["To Min"].default_value = 0.0
        remap.inputs["To Max"].default_value = 1.0
    else:
        remap.inputs["From Min"].default_value = near
        remap.inputs["From Max"].default_value = far
        remap.inputs["To Min"].default_value = 1.0
        remap.inputs["To Max"].default_value = 0.0
    # cancel the sRGB display curve so the saved PNG is linear in depth
    degamma = nt.nodes.new("ShaderNodeMath")
    degamma.operation = "POWER"
    degamma.inputs[1].default_value = 2.2
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(depth_socket, remap.inputs["Value"])
    nt.links.new(remap.outputs["Result"], degamma.inputs[0])
    nt.links.new(degamma.outputs["Value"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def set_engine(r, names):
    for name in names:
        try:
            r.engine = name
            return
        except TypeError:
            continue
    raise RuntimeError(f"none of the render engines {names} are available")


def render_sequence(scene, times, folder):
    os.makedirs(folder, exist_ok=True)
    for i, t in enumerate(times, start=1):
        set_time(scene, t)
        scene.render.filepath = os.path.join(folder, f"{i:04d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    if args.test_scene or not bpy.data.filepath:
        build_test_scene(args.frames)
    scene = bpy.context.scene

    if args.camera:
        scene.camera = bpy.data.objects[args.camera]
    cam = scene.camera
    if cam is None:
        sys.exit("ERROR: scene has no camera")

    times = scene_times(scene, args.frames, args.start)
    near, far = auto_near_far(scene, cam, times)
    near = args.near if args.near is not None else near
    far = args.far if args.far is not None else far

    # everything below changes the scene in memory only; the .blend is never saved
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = args.width, args.height, 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("WanDepthWorld")
    world.color = (0, 0, 0)
    if world.node_tree:  # Blender 5 worlds come with a gray Background node
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (0, 0, 0, 1)
                node.inputs["Strength"].default_value = 0
    scene.world = world

    view_layer = bpy.context.view_layer
    set_engine(r, ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"))
    view_layer.material_override = depth_material(near, far, args.depth_curve)
    render_sequence(scene, times, os.path.join(out_dir, "depth"))

    if not args.no_preview:
        view_layer.material_override = None
        r.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "SINGLE"
        shading.single_color = (0.6, 0.6, 0.6)
        render_sequence(scene, times, os.path.join(out_dir, "gray"))

    info = {
        "frames": args.frames, "fps": WAN_FPS, "near": near, "far": far, "depth_curve": args.depth_curve,
        "scene_fps": r.fps / r.fps_base, "scene_frames": [times[0], times[-1]],
        "camera": cam.name, "blend": bpy.data.filepath or "(test scene)",
    }
    with open(os.path.join(out_dir, "depth_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("DEPTH_DONE", json.dumps(info))


if __name__ == "__main__":
    main()
