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
    # Single-image input
    image_path: StringProperty(
        name="Image",
        description="Select an image to upload",
        subtype='FILE_PATH'
    )
    # Multi-view inputs
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
    # Generation settings
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
    server_status: StringProperty(name="Server Status", default="Not checked")
    server_model_path: StringProperty(name="Server Model", default="Unknown")
    server_subfolder: StringProperty(name="Server Subfolder", default="Unknown")
    server_enable_flashvdm: StringProperty(name="Server FlashVDM", default="Unknown")
    server_enable_tex: StringProperty(name="Server Texture", default="Unknown")
    server_enable_t23d: StringProperty(name="Server Text2Img", default="Unknown")


class Hunyuan3DSavePresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.save_preset"
    bl_label = "Save Settings"
    bl_description = "Save the current Hunyuan3D settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.preset_name.strip()
        if not name:
            self.report({'WARNING'}, "Enter a preset name first.")
            return {'CANCELLED'}

        data = _load_presets()
        data[name] = {
            "use_multiview": props.use_multiview,
            "auto_remove_background": props.auto_remove_background,
            "octree_resolution": props.octree_resolution,
            "num_inference_steps": props.num_inference_steps,
            "guidance_scale": props.guidance_scale,
            "texture": props.texture,
            "api_url": props.api_url,
        }
        _save_presets(data)
        props.selected_preset = name
        self.report({'INFO'}, f"Saved settings: {name}")
        return {'FINISHED'}


class Hunyuan3DLoadPresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.load_preset"
    bl_label = "Load Settings"
    bl_description = "Load the selected Hunyuan3D settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.selected_preset
        data = _load_presets()
        if not name or name not in data:
            self.report({'WARNING'}, "Select a saved setting set first.")
            return {'CANCELLED'}

        preset = data[name]
        props.use_multiview = preset.get("use_multiview", props.use_multiview)
        props.auto_remove_background = preset.get("auto_remove_background", props.auto_remove_background)
        props.octree_resolution = preset.get("octree_resolution", props.octree_resolution)
        props.num_inference_steps = preset.get("num_inference_steps", props.num_inference_steps)
        props.guidance_scale = preset.get("guidance_scale", props.guidance_scale)
        props.texture = preset.get("texture", props.texture)
        props.api_url = preset.get("api_url", props.api_url)
        self.report({'INFO'}, f"Loaded settings: {name}")
        return {'FINISHED'}


class Hunyuan3DDeletePresetOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.delete_preset"
    bl_label = "Delete Settings"
    bl_description = "Delete the selected Hunyuan3D settings"

    def execute(self, context):
        props = context.scene.gen_3d_props
        name = props.selected_preset
        data = _load_presets()
        if not name or name not in data:
            self.report({'WARNING'}, "Select a saved setting set first.")
            return {'CANCELLED'}

        del data[name]
        _save_presets(data)
        props.selected_preset = ""
        self.report({'INFO'}, f"Deleted settings: {name}")
        return {'FINISHED'}


class Hunyuan3DRefreshGPUOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.refresh_gpu"
    bl_label = "Refresh GPU"
    bl_description = "Refresh GPU usage information"

    def execute(self, context):
        stats = _poll_gpu_stats(force=True)
        props = context.scene.gen_3d_props
        props.gpu_name = stats["name"]
        props.gpu_memory = stats["memory"]
        props.gpu_utilization = stats["utilization"]
        props.gpu_status_message = stats["status"]
        return {'FINISHED'}


class Hunyuan3DRefreshServerOperator(bpy.types.Operator):
    bl_idname = "hunyuan3d.refresh_server"
    bl_label = "Refresh Server"
    bl_description = "Refresh active server information"

    def execute(self, context):
        props = context.scene.gen_3d_props
        stats = _poll_server_info(props.api_url, force=True)
        props.server_status = stats["status"]
        props.server_model_path = stats["model_path"]
        props.server_subfolder = stats["subfolder"]
        props.server_enable_flashvdm = stats["enable_flashvdm"]
        props.server_enable_tex = stats["enable_tex"]
        props.server_enable_t23d = stats["enable_t23d"]
        return {'FINISHED'}


class Hunyuan3DOperator(bpy.types.Operator):
    bl_idname = "object.generate_3d"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model from text description, an image or a selected mesh"

    job_id = ''
    prompt = ""
    api_url = ""
    image_path = ""
    use_multiview = False
    mv_image_front = ""
    mv_image_back = ""
    mv_image_left = ""
    mv_image_right = ""
    octree_resolution = 256
    num_inference_steps = 20
    guidance_scale = 5.5
    auto_remove_background = False
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
            print("Threaded task completed")
            self.task_finished = False
            props = context.scene.gen_3d_props
            props.is_processing = False

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        # 启动线程
        props = context.scene.gen_3d_props
        self.prompt = props.prompt
        self.api_url = props.api_url
        self.image_path = props.image_path
        self.use_multiview = props.use_multiview
        self.mv_image_front = props.mv_image_front
        self.mv_image_back = props.mv_image_back
        self.mv_image_left = props.mv_image_left
        self.mv_image_right = props.mv_image_right
        self.octree_resolution = props.octree_resolution
        self.num_inference_steps = props.num_inference_steps
        self.guidance_scale = props.guidance_scale
        self.auto_remove_background = props.auto_remove_background
        self.texture = props.texture

        has_mv = any([self.mv_image_front, self.mv_image_back, self.mv_image_left, self.mv_image_right])
        if self.use_multiview:
            if not has_mv:
                self.report({'WARNING'}, "Please select at least one multiview image.")
                return {'FINISHED'}
        else:
            if self.prompt == "" and self.image_path == "":
                self.report({'WARNING'}, "Please enter some text or select an image first.")
                return {'FINISHED'}

        # 保存选中的 mesh 对象引用
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                self.selected_mesh = obj
                break

        if self.selected_mesh:
            temp_glb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            temp_glb_file.close()
            bpy.ops.export_scene.gltf(filepath=temp_glb_file.name, use_selection=True)
            with open(temp_glb_file.name, "rb") as file:
                mesh_data = file.read()
            mesh_b64_str = base64.b64encode(mesh_data).decode()
            os.unlink(temp_glb_file.name)
            self.selected_mesh_base64 = mesh_b64_str

        props.is_processing = True

        self.image_path = self._resolve_path(self.image_path)
        self.mv_image_front = self._resolve_path(self.mv_image_front)
        self.mv_image_back = self._resolve_path(self.mv_image_back)
        self.mv_image_left = self._resolve_path(self.mv_image_left)
        self.mv_image_right = self._resolve_path(self.mv_image_right)

        if self.selected_mesh and self.texture:
            props.status_message = "Texturing Selected Mesh...\n" \
                                   "This may take several minutes depending \n on your GPU power."
        else:
            mesh_type = 'Textured Mesh' if self.texture else 'White Mesh'
            if self.use_multiview:
                prompt_type = 'MultiView Images'
            else:
                prompt_type = 'Text Prompt' if self.prompt else 'Image'
            props.status_message = f"Generating {mesh_type} with {prompt_type}...\n" \
                                   "This may take several minutes depending \n on your GPU power."

        self.thread = threading.Thread(target=self.generate_model, args=[context])
        self.thread.start()

        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def generate_model(self, context):
        self.report({'INFO'}, f"Generation Start")
        base_url = self.api_url.rstrip('/')

        try:
            base_payload = {
                "octree_resolution": self.octree_resolution,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "auto_remove_background": self.auto_remove_background,
                "texture": self.texture
            }

            if self.use_multiview:
                mv_payload = self._build_mv_payload()
                if not mv_payload:
                    raise Exception("No multiview images found.")
            else:
                mv_payload = {}

            if self.selected_mesh_base64 and self.texture:
                payload = dict(base_payload)
                payload["mesh"] = self.selected_mesh_base64
                if self.use_multiview and mv_payload:
                    self.report({'INFO'}, "Post Texturing with MultiView Images")
                    payload.update(mv_payload)
                elif self.image_path:
                    self.report({'INFO'}, "Post Texturing with Image")
                    payload["image"] = self._encode_image_file(self.image_path)
                else:
                    self.report({'INFO'}, "Post Texturing with Text")
                    payload["text"] = self.prompt
                response = requests.post(f"{base_url}/generate", json=payload)
            else:
                payload = dict(base_payload)
                if self.use_multiview and mv_payload:
                    self.report({'INFO'}, "Post Start MultiView to 3D")
                    payload.update(mv_payload)
                elif self.image_path:
                    self.report({'INFO'}, "Post Start Image to 3D")
                    payload["image"] = self._encode_image_file(self.image_path)
                else:
                    self.report({'INFO'}, "Post Start Text to 3D")
                    payload["text"] = self.prompt
                response = requests.post(f"{base_url}/generate", json=payload)
            self.report({'INFO'}, f"Post Done")
            props = context.scene.gen_3d_props

            if response.status_code != 200:
                self.report({'ERROR'}, f"Generation failed: {response.text}")
                return

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            temp_file.write(response.content)
            temp_file.close()

            # Import the GLB file in the main thread
            def import_handler():
                bpy.ops.import_scene.gltf(filepath=temp_file.name)
                os.unlink(temp_file.name)

                # 获取新导入的 mesh
                new_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
                if new_obj and self.selected_mesh and self.texture:
                    # 应用选中 mesh 的位置、旋转和缩放
                    new_obj.location = self.selected_mesh.location
                    new_obj.rotation_euler = self.selected_mesh.rotation_euler
                    new_obj.scale = self.selected_mesh.scale

                    # 隐藏原来的 mesh
                    self.selected_mesh.hide_set(True)
                    self.selected_mesh.hide_render = True

                return None

            bpy.app.timers.register(import_handler)

        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")

        finally:
            self.task_finished = True
            props = context.scene.gen_3d_props
            props.is_processing = False
            self.selected_mesh_base64 = ""


class HUNYUAN3D_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Nymphs3D'
    bl_label = 'Nymphs3D'

    def draw(self, context):
        layout = self.layout
        props = context.scene.gen_3d_props

        layout.prop(props, "api_url")
        layout.prop(props, "use_multiview")
        layout.prop(props, "prompt")

        if props.use_multiview:
            box = layout.box()
            box.label(text="MultiView Images")
            box.prop(props, "mv_image_front")
            box.prop(props, "mv_image_back")
            box.prop(props, "mv_image_left")
            box.prop(props, "mv_image_right")
        else:
            layout.prop(props, "image_path")

        layout.prop(props, "auto_remove_background")
        layout.prop(props, "octree_resolution")
        layout.prop(props, "num_inference_steps")
        layout.prop(props, "guidance_scale")
        layout.prop(props, "texture")

        box = layout.box()
        box.prop(
            props,
            "show_saved_setups",
            text="Presets",
            icon='TRIA_DOWN' if props.show_saved_setups else 'TRIA_RIGHT',
            emboss=False,
        )
        if props.show_saved_setups:
            box.prop(props, "preset_name", text="Save As")
            row = box.row(align=True)
            row.operator("hunyuan3d.save_preset", text="Save Current Setup")
            box.prop(props, "selected_preset", text="Load Setup")
            row = box.row(align=True)
            row.operator("hunyuan3d.load_preset", text="Load Setup")
            row.operator("hunyuan3d.delete_preset", text="Delete Setup")

        box = layout.box()
        box.prop(
            props,
            "show_server_info",
            text="Server Info",
            icon='TRIA_DOWN' if props.show_server_info else 'TRIA_RIGHT',
            emboss=False,
        )
        if props.show_server_info:
            box.label(text=f"Status: {props.server_status}")
            box.label(text=f"Model: {props.server_model_path}")
            box.label(text=f"Subfolder: {props.server_subfolder}")
            box.label(text=f"FlashVDM: {props.server_enable_flashvdm}")
            box.label(text=f"Texture: {props.server_enable_tex}")
            box.label(text=f"Text2Img: {props.server_enable_t23d}")
            box.operator("hunyuan3d.refresh_server", text="Refresh Server")

        box = layout.box()
        box.prop(
            props,
            "show_gpu_status",
            text="GPU Status",
            icon='TRIA_DOWN' if props.show_gpu_status else 'TRIA_RIGHT',
            emboss=False,
        )
        if props.show_gpu_status:
            box.label(text=f"GPU: {props.gpu_name}")
            box.label(text=f"VRAM: {props.gpu_memory}")
            box.label(text=f"Load: {props.gpu_utilization}")
            box.label(text=f"Status: {props.gpu_status_message}")
            box.operator("hunyuan3d.refresh_gpu", text="Refresh GPU")

        row = layout.row()
        row.enabled = not props.is_processing
        row.operator("object.generate_3d")

        if props.is_processing:
            if props.status_message:
                for line in props.status_message.split("\n"):
                    layout.label(text=line)
            else:
                layout.label("Processing...")


classes = (
    Hunyuan3DProperties,
    Hunyuan3DSavePresetOperator,
    Hunyuan3DLoadPresetOperator,
    Hunyuan3DDeletePresetOperator,
    Hunyuan3DRefreshGPUOperator,
    Hunyuan3DRefreshServerOperator,
    Hunyuan3DOperator,
    HUNYUAN3D_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gen_3d_props = bpy.props.PointerProperty(type=Hunyuan3DProperties)
    if not bpy.app.timers.is_registered(_gpu_status_timer):
        bpy.app.timers.register(_gpu_status_timer, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_gpu_status_timer):
        bpy.app.timers.unregister(_gpu_status_timer)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gen_3d_props


if __name__ == "__main__":
    register()
