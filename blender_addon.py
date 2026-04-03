# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

bl_info = {
    "name": "Nymphs3D",
    "author": "Tencent Hunyuan3D",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Nymphs3D",
    "description": "Generate 3D models from images for the Nymphs3D workflow",
    "category": "3D View",
}
import base64
import json
import os
import subprocess
import tempfile
import threading
import time

import bpy
import requests
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty


PRESET_PATH = os.path.join(os.path.expanduser("~"), ".hunyuan3d_blender_presets.json")
GPU_REFRESH_SECONDS = 2.0
_last_gpu_poll = 0.0
_gpu_cache = {
    "name": "Unknown",
    "memory": "Unavailable",
    "utilization": "Unavailable",
    "status": "Not checked",
}
_server_cache = {
    "status": "Not checked",
    "model_path": "Unknown",
    "subfolder": "Unknown",
    "enable_flashvdm": "Unknown",
    "enable_tex": "Unknown",
    "enable_t23d": "Unknown",
}


def _load_presets():
    if not os.path.exists(PRESET_PATH):
        return {}
    try:
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_presets(data):
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _preset_items(self, context):
    items = [("", "Select Saved Settings", "Load a saved setting set")]
    for name in sorted(_load_presets()):
        items.append((name, name, f"Load saved settings: {name}"))
    return items


def _poll_gpu_stats(force=False):
    global _last_gpu_poll, _gpu_cache
    now = time.time()
    if not force and now - _last_gpu_poll < GPU_REFRESH_SECONDS:
        return _gpu_cache

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        name, mem_total, mem_used, util = [part.strip() for part in line.split(",")]
        _gpu_cache = {
            "name": name,
            "memory": f"{mem_used} / {mem_total} MB",
            "utilization": f"{util}%",
            "status": "Connected",
        }
    except Exception:
        _gpu_cache = {
            "name": "Unavailable",
            "memory": "Unavailable",
            "utilization": "Unavailable",
            "status": "nvidia-smi not available",
        }
    _last_gpu_poll = now
    return _gpu_cache


def _gpu_status_timer():
    stats = _poll_gpu_stats(force=True)
    for scene in bpy.data.scenes:
        props = getattr(scene, "gen_3d_props", None)
        if props is None:
            continue
        props.gpu_name = stats["name"]
        props.gpu_memory = stats["memory"]
        props.gpu_utilization = stats["utilization"]
        props.gpu_status_message = stats["status"]
    return GPU_REFRESH_SECONDS


def _poll_server_info(api_url, force=False):
    global _server_cache
    try:
        response = requests.get(f"{api_url.rstrip('/')}/server_info", timeout=2)
        response.raise_for_status()
        data = response.json()
        _server_cache = {
            "status": data.get("status", "ok"),
            "model_path": data.get("model_path", "Unknown"),
            "subfolder": data.get("subfolder", "Unknown"),
            "enable_flashvdm": str(data.get("enable_flashvdm", "Unknown")),
            "enable_tex": str(data.get("enable_tex", "Unknown")),
            "enable_t23d": str(data.get("enable_t23d", "Unknown")),
        }
    except Exception:
        _server_cache = {
            "status": "Unavailable",
            "model_path": "Unknown",
            "subfolder": "Unknown",
            "enable_flashvdm": "Unknown",
            "enable_tex": "Unknown",
            "enable_t23d": "Unknown",
        }
    return _server_cache


class Hunyuan3DProperties(bpy.types.PropertyGroup):
    prompt: StringProperty(
        name="Text Prompt",
        description="Describe what you want to generate",
        default=""
    )
    api_url: StringProperty(
        name="API URL",
        description="URL of the Text-to-3D API service",
        default="http://localhost:8080"
    )
    is_processing: BoolProperty(
        name="Processing",
        default=False
    )
    job_id: StringProperty(
        name="Job ID",
        default=""
    )
    status_message: StringProperty(
        name="Status Message",
        default=""
    )
    image_path: StringProperty(
        name="Image",
        description="Select an image to upload",
        subtype='FILE_PATH'
    )
    use_multiview: BoolProperty(
        name="Use MultiView",
        description="Use Front/Back/Left/Right image inputs",
        default=False
    )
    mv_image_front: StringProperty(
        name="Front",
        description="Front view image",
        subtype='FILE_PATH'
    )
    mv_image_back: StringProperty(
        name="Back",
        description="Back view image",
        subtype='FILE_PATH'
    )
    mv_image_left: StringProperty(
        name="Left",
        description="Left view image",
        subtype='FILE_PATH'
    )
    mv_image_right: StringProperty(
        name="Right",
        description="Right view image",
        subtype='FILE_PATH'
    )
    auto_remove_background: BoolProperty(
        name="Auto Remove Background",
        description="Automatically isolate the subject from the background before generation",
        default=False,
    )
    octree_resolution: IntProperty(
        name="Mesh Detail",
        description="Higher values keep more shape detail but take longer and use more GPU memory",
        default=256,
        min=128,
        max=512,
    )
    num_inference_steps: IntProperty(
        name="Detail Passes",
        description="Higher values spend longer refining the result",
        default=20,
        min=20,
        max=100
    )
    guidance_scale: FloatProperty(
        name="Reference Strength",
        description="How strongly the result should follow the input images",
        default=5.5,
        min=1.0,
        max=10.0
    )
    texture: BoolProperty(
        name="Generate Texture",
        description="Whether to generate texture for the 3D model",
        default=False
    )
    preset_name: StringProperty(
        name="Preset Name",
        description="Name to use when saving the current settings",
        default="",
    )
    selected_preset: EnumProperty(
        name="Saved Settings",
        description="Load a previously saved setting set",
        items=_preset_items,
    )
    show_saved_setups: BoolProperty(name="Show Saved Setups", default=False)
    show_server_info: BoolProperty(name="Show Server Info", default=False)
    show_gpu_status: BoolProperty(name="Show GPU Status", default=False)
    gpu_name: StringProperty(name="GPU Name", default="Unknown")
    gpu_memory: StringProperty(name="GPU Memory", default="Unavailable")
    gpu_utilization: StringProperty(name="GPU Utilization", default="Unavailable")
    gpu_status_message: StringProperty(name="GPU Status", default="Not checked")


class Hunyuan3DOperator(bpy.types.Operator):
    bl_idname = "object.generate_3d"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model from text, image, multiview images or selected mesh"

    job_id = ''
    prompt = ""
    api_url = ""
    image_path = ""
    use_multiview = False
    mv_image_front = ""
    mv_image_back = ""
    mv_image_left = ""
    mv_image_right = ""
    auto_remove_background = False
    octree_resolution = 256
    num_inference_steps = 20
    guidance_scale = 5.5
    texture = False
    selected_mesh_base64 = ""
    selected_mesh = None

    thread = None
    task_finished = False

    def _resolve_path(self, path):
        if not path:
            return ""
        if path.startswith('//'):
            blend_file_dir = os.path.dirname(bpy.data.filepath)
            path = os.path.join(blend_file_dir, path[2:])
        return path

    def _encode_image_file(self, path):
        if not path:
            return None
        path = self._resolve_path(path)
        if not os.path.exists(path):
            raise Exception(f'Image path does not exist {path}')
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()

    def _build_mv_payload(self):
        payload = {}
        front = self._encode_image_file(self.mv_image_front) if self.mv_image_front else None
        back = self._encode_image_file(self.mv_image_back) if self.mv_image_back else None
        left = self._encode_image_file(self.mv_image_left) if self.mv_image_left else None
        right = self._encode_image_file(self.mv_image_right) if self.mv_image_right else None
        if front:
            payload["mv_image_front"] = front
        if back:
            payload["mv_image_back"] = back
        if left:
            payload["mv_image_left"] = left
        if right:
            payload["mv_image_right"] = right
        return payload

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}

        if self.task_finished:
            self.task_finished = False
            props = context.scene.gen_3d_props
            props.is_processing = False

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        props = context.scene.gen_3d_props
        self.prompt = props.prompt
        self.api_url = props.api_url
        self.image_path = props.image_path
        self.use_multiview = props.use_multiview
        self.mv_image_front = props.mv_image_front
        self.mv_image_back = props.mv_image_back
        self.mv_image_left = props.mv_image_left
        self.mv_image_right = props.mv_image_right
        self.auto_remove_background = props.auto_remove_background
        self.octree_resolution = props.octree_resolution
        self.num_inference_steps = props.num_inference_steps
        self.guidance_scale = props.guidance_scale
        self.texture = props.texture

        has_mv = any([self.mv_image_front, self.mv_image_back, self.mv_image_left, self.mv_image_right])
        if self.use_multiview:
            if not has_mv:
                self.report({'WARNING'}, "Please select at least one multiview image.")
                return {'FINISHED'}
        else:
            if self.prompt == "" and self.image_path == "":
                self.report({'WARNING'}, "Please enter text or select an image first.")
                return {'FINISHED'}

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                self.selected_mesh = obj
                break

        props.is_processing = True
        self.thread = threading.Thread(target=self.generate_model)
        self.thread.start()

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def generate_model(self):
        try:
            payload = {
                "octree_resolution": self.octree_resolution,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "texture": self.texture,
                "auto_remove_background": self.auto_remove_background,
            }

            if self.use_multiview:
                payload.update(self._build_mv_payload())
            elif self.image_path:
                payload["image"] = self._encode_image_file(self.image_path)
            elif self.prompt:
                payload["text"] = self.prompt

            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            self.job_id = result["uid"]
            bpy.app.timers.register(self.check_status)
        except Exception as e:
            print(f"Generation failed: {e}")
            self.task_finished = True

    def check_status(self):
        try:
            response = requests.get(f"{self.api_url}/status/{self.job_id}", timeout=10)
            response.raise_for_status()
            status = response.json()

            if status["status"] == "completed":
                file_response = requests.get(f"{self.api_url}/file/{self.job_id}", timeout=120)
                file_response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
                    tmp_file.write(file_response.content)
                    tmp_path = tmp_file.name
                bpy.ops.import_scene.gltf(filepath=tmp_path)
                self.task_finished = True
                return None

            if status["status"] == "failed":
                print("Generation failed on server")
                self.task_finished = True
                return None

            return 2.0
        except Exception as e:
            print(f"Status check failed: {e}")
            self.task_finished = True
            return None


class Hunyuan3DSavePresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.save_preset"
    bl_label = "Save Current Settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.preset_name.strip()
        if not name:
            self.report({'WARNING'}, "Preset name is required.")
            return {'CANCELLED'}

        data = _load_presets()
        data[name] = {
            "api_url": props.api_url,
            "use_multiview": props.use_multiview,
            "auto_remove_background": props.auto_remove_background,
            "octree_resolution": props.octree_resolution,
            "num_inference_steps": props.num_inference_steps,
            "guidance_scale": props.guidance_scale,
            "texture": props.texture,
        }
        _save_presets(data)
        self.report({'INFO'}, f"Saved preset: {name}")
        return {'FINISHED'}


class Hunyuan3DLoadPresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.load_preset"
    bl_label = "Load Selected Settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.selected_preset
        data = _load_presets()
        if not name or name not in data:
            self.report({'WARNING'}, "Select a saved preset first.")
            return {'CANCELLED'}

        preset = data[name]
        props.api_url = preset.get("api_url", props.api_url)
        props.use_multiview = preset.get("use_multiview", props.use_multiview)
        props.auto_remove_background = preset.get("auto_remove_background", props.auto_remove_background)
        props.octree_resolution = preset.get("octree_resolution", props.octree_resolution)
        props.num_inference_steps = preset.get("num_inference_steps", props.num_inference_steps)
        props.guidance_scale = preset.get("guidance_scale", props.guidance_scale)
        props.texture = preset.get("texture", props.texture)
        self.report({'INFO'}, f"Loaded preset: {name}")
        return {'FINISHED'}


class Hunyuan3DDeletePresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.delete_preset"
    bl_label = "Delete Selected Settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.selected_preset
        data = _load_presets()
        if not name or name not in data:
            self.report({'WARNING'}, "Select a saved preset first.")
            return {'CANCELLED'}

        del data[name]
        _save_presets(data)
        props.selected_preset = ""
        self.report({'INFO'}, f"Deleted preset: {name}")
        return {'FINISHED'}


class Hunyuan3DPanel(bpy.types.Panel):
    bl_label = "Nymphs3D"
    bl_idname = "VIEW3D_PT_hunyuan3d_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Nymphs3D'

    def draw(self, context):
        layout = self.layout
        props = context.scene.gen_3d_props

        box = layout.box()
        row = box.row()
        row.prop(props, "show_saved_setups", icon='TRIA_DOWN' if props.show_saved_setups else 'TRIA_RIGHT', emboss=False)
        row.label(text="Saved Setups")
        if props.show_saved_setups:
            box.prop(props, "preset_name")
            box.operator("hunyuan3d.save_preset", icon='ADD')
            box.prop(props, "selected_preset")
            row = box.row(align=True)
            row.operator("hunyuan3d.load_preset", icon='IMPORT')
            row.operator("hunyuan3d.delete_preset", icon='TRASH')

        layout.prop(props, "api_url")
        layout.prop(props, "use_multiview")

        if props.use_multiview:
            layout.prop(props, "mv_image_front")
            layout.prop(props, "mv_image_back")
            layout.prop(props, "mv_image_left")
            layout.prop(props, "mv_image_right")
        else:
            layout.prop(props, "prompt")
            layout.prop(props, "image_path")

        layout.prop(props, "auto_remove_background")
        layout.prop(props, "octree_resolution")
        layout.prop(props, "num_inference_steps")
        layout.prop(props, "guidance_scale")
        layout.prop(props, "texture")

        row = layout.row()
        row.enabled = not props.is_processing
        row.operator("object.generate_3d", icon='MESH_MONKEY')

        server_box = layout.box()
        row = server_box.row()
        row.prop(props, "show_server_info", icon='TRIA_DOWN' if props.show_server_info else 'TRIA_RIGHT', emboss=False)
        row.label(text="Server Info")
        if props.show_server_info:
            info = _poll_server_info(props.api_url)
            server_box.label(text=f"Status: {info['status']}")
            server_box.label(text=f"Model: {info['model_path']}")
            server_box.label(text=f"Subfolder: {info['subfolder']}")
            server_box.label(text=f"FlashVDM: {info['enable_flashvdm']}")
            server_box.label(text=f"Texture: {info['enable_tex']}")
            server_box.label(text=f"Text-to-3D: {info['enable_t23d']}")

        gpu_box = layout.box()
        row = gpu_box.row()
        row.prop(props, "show_gpu_status", icon='TRIA_DOWN' if props.show_gpu_status else 'TRIA_RIGHT', emboss=False)
        row.label(text="GPU Status")
        if props.show_gpu_status:
            gpu_box.label(text=f"GPU: {props.gpu_name}")
            gpu_box.label(text=f"Memory: {props.gpu_memory}")
            gpu_box.label(text=f"Utilization: {props.gpu_utilization}")
            gpu_box.label(text=f"State: {props.gpu_status_message}")


classes = (
    Hunyuan3DProperties,
    Hunyuan3DOperator,
    Hunyuan3DSavePresetOperator,
    Hunyuan3DLoadPresetOperator,
    Hunyuan3DDeletePresetOperator,
    Hunyuan3DPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gen_3d_props = bpy.props.PointerProperty(type=Hunyuan3DProperties)
    bpy.app.timers.register(_gpu_status_timer, persistent=True)


def unregister():
    if hasattr(bpy.types.Scene, "gen_3d_props"):
        del bpy.types.Scene.gen_3d_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
