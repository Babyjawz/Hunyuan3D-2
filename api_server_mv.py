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

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO
import time

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from progress_state import snapshot as progress_snapshot, update as progress_update, reset as progress_reset

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    # Prevent Python logging internals from recursively writing
    # "--- Logging error ---" to redirected stderr when a console
    # handler hits a broken pipe under external launchers.
    logging.raiseExceptions = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)
    if root_logger.handlers:
        root_logger.handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO, terminal=sys.__stdout__)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR, terminal=sys.__stderr__)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        # Remove default console handlers so logging never writes directly
        # to a launcher-managed pipe. StreamToLogger handles terminal output.
        for existing in list(root_logger.handlers):
            root_logger.removeHandler(existing)
        root_logger.addHandler(handler)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO, terminal=None):
        self.terminal = terminal or sys.__stdout__
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        try:
            if self.terminal and buf:
                self.terminal.write(buf)
            temp_linebuf = self.linebuf + buf
            self.linebuf = ''
            for line in temp_linebuf.splitlines(True):
                if line[-1] == '\n':
                    self.logger.log(self.log_level, line.rstrip())
                else:
                    self.linebuf += line
        except BrokenPipeError:
            pass

    def flush(self):
        try:
            if self.terminal:
                self.terminal.flush()
            if self.linebuf != '':
                self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''
        except BrokenPipeError:
            self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def emit_stage(stage):
    progress_set_stage(stage)
    print(f"STAGE: {stage}", flush=True)


def emit_progress(name, current, total):
    percent = max(0.0, min(100.0, (float(current) / float(total)) * 100.0)) if total else None
    progress_update(
        status="processing",
        stage=name.strip().replace(" ", "_").lower(),
        detail=None,
        progress_current=current,
        progress_total=total,
        progress_percent=percent,
        message=None,
    )
    print(f"PROGRESS: {name} {current}/{total}", flush=True)


def texture_prompt_images(image):
    if not isinstance(image, dict):
        return image

    # The texture pipeline accepts either a single PIL image or a list of PIL
    # images. Keep a stable multiview order that matches the repo examples.
    ordered_views = ("front", "left", "back", "right")
    images = [image[view] for view in ordered_views if view in image]
    return images or image


def progress_set_stage(stage, detail=None, status="processing", current=None, total=None, percent=None, message=None):
    friendly_detail = detail if detail is not None else stage.replace("_", " ").title()
    progress_update(
        status=status,
        stage=stage,
        detail=friendly_detail,
        progress_current=current,
        progress_total=total,
        progress_percent=percent,
        message=message,
    )


def emit_texture_progress(step_index, total_steps, label, *, started=False, elapsed=None):
    current = max(0, step_index - (0 if elapsed is not None else 1))
    percent = max(0.0, min(100.0, (float(current) / float(total_steps)) * 100.0)) if total_steps else None
    detail = label if started or elapsed is None else f"{label} ({elapsed:.1f}s)"
    progress_set_stage(
        "generating_texture",
        detail=detail,
        current=current,
        total=total_steps,
        percent=percent,
    )
    if elapsed is not None:
        emit_progress("texture", current, total_steps)
        logger.info("Texture timing %s/%s: %s", current, total_steps, detail)
    else:
        logger.info("Texture step %s/%s started: %s", step_index, total_steps, label)


def validate_request_options(params, worker):
    if params.get("texture", False) and not hasattr(worker, "pipeline_tex"):
        raise ValueError("Texture generation was requested, but the server was not started with --enable_tex.")

    if params.get("text") and worker.pipeline_t2i is None:
        raise ValueError("Text-to-3D was requested, but the server was not started with --enable_t23d.")

    if "mesh" in params and not (
        params.get("image")
        or params.get("text")
        or any(k in params for k in ("mv_image_front", "mv_image_back", "mv_image_left", "mv_image_right"))
    ):
        raise ValueError("Mesh texturing requires a guiding image, multiview images, or a text prompt.")


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mv-turbo',
                 device='cuda',
                 enable_tex=False,
                 enable_flashvdm=False):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        emit_stage("loading_shape_model")
        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        emit_progress("startup", 1, 3)
        if enable_flashvdm:
            self.pipeline.enable_flashvdm(mc_algo='mc')
        self.pipeline_t2i = None
        if args.enable_t23d:
            emit_stage("loading_text_bridge")
            self.pipeline_t2i = HunyuanDiTPipeline(
                'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                device=device
            )
            emit_progress("startup", 2, 3)
            emit_stage("text_bridge_ready")
        else:
            emit_progress("startup", 2, 3)
        if enable_tex:
            emit_stage("loading_texture_pipeline")
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)
        emit_progress("startup", 3, 3)
        emit_stage("server_boot_complete")

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        progress_reset()
        progress_update(status="processing", uid=str(uid), stage="request_received", detail="Request received", message=None)
        validate_request_options(params, self)
        emit_stage("request_received")
        if any(k in params for k in ('mv_image_front', 'mv_image_back', 'mv_image_left', 'mv_image_right')):
            emit_stage("loading_multiview_images")
            image = {}
            if 'mv_image_front' in params:
                image['front'] = load_image_from_base64(params['mv_image_front'])
            if 'mv_image_back' in params:
                image['back'] = load_image_from_base64(params['mv_image_back'])
            if 'mv_image_left' in params:
                image['left'] = load_image_from_base64(params['mv_image_left'])
            if 'mv_image_right' in params:
                image['right'] = load_image_from_base64(params['mv_image_right'])
        elif 'image' in params:
            emit_stage("loading_input_image")
            image = params["image"]
            image = load_image_from_base64(image)
            if not isinstance(image, dict):
                image = {'front': image}
        else:
            if 'text' in params:
                text = params["text"]
                if self.pipeline_t2i is None:
                    raise ValueError("Text-to-3D is disabled. Start the server with --enable_t23d")
                emit_stage("generating_text_image")
                image = self.pipeline_t2i(text)
                emit_stage("text_image_ready")
                if not isinstance(image, dict):
                    image = {'front': image}
            else:
                raise ValueError("No input image or text provided")

        if params.get("auto_remove_background", True):
            emit_stage("preparing_input")
            if isinstance(image, dict):
                image = {k: self.rembg(v) for k, v in image.items()}
            else:
                image = self.rembg(image)
        params['image'] = image

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            seed = params.get("seed", 1234)
            params['generator'] = torch.Generator(self.device).manual_seed(seed)
            params['octree_resolution'] = params.get("octree_resolution", 128)
            params['num_inference_steps'] = params.get("num_inference_steps", 5)
            params['guidance_scale'] = params.get('guidance_scale', 5.0)
            params['mc_algo'] = 'mc'
            import time
            start_time = time.time()
            emit_stage("sampling_shape")
            mesh = self.pipeline(**params)[0]
            emit_stage("shape_ready")
            logger.info("--- %s seconds ---" % (time.time() - start_time))

        if params.get('texture', False):
            emit_stage("generating_texture")
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            texture_started = time.time()
            mesh = self.pipeline_tex(
                mesh,
                texture_prompt_images(image),
                progress_callback=emit_texture_progress,
            )
            logger.info("Texture pipeline total: %.1fs", time.time() - texture_started)
            emit_stage("texture_ready")

        type = params.get('type', 'glb')
        emit_stage("exporting_mesh")
        with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{type}')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        progress_update(
            status="completed",
            uid=str(uid),
            stage="job_complete",
            detail="Completed",
            progress_current=None,
            progress_total=None,
            progress_percent=100.0,
            message=None,
        )
        emit_stage("job_complete")
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = await run_in_threadpool(worker.generate, uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        progress_update(status="failed", uid=str(uid), stage="failed", detail=str(e), message=str(e))
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        progress_update(status="failed", uid=str(uid), stage="failed", detail=str(e), message=str(e))
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        progress_update(status="failed", uid=str(uid), stage="failed", detail=str(e), message=str(e))
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    progress_reset()
    progress_update(status="processing", uid=str(uid), stage="request_received", detail="Request received", message=None)
    threading.Thread(target=worker.generate, args=(uid, params,), daemon=True).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/server_info")
async def server_info():
    return JSONResponse(
        {
            "status": "ok",
            "model_path": args.model_path,
            "subfolder": args.subfolder,
            "enable_flashvdm": args.enable_flashvdm,
            "enable_tex": args.enable_tex,
            "enable_t23d": args.enable_t23d,
            "device": args.device,
            "port": args.port,
        },
        status_code=200,
    )


@app.get("/active_task")
async def active_task():
    return JSONResponse(progress_snapshot(), status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mv-turbo')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(
        model_path=args.model_path,
        tex_model_path=args.tex_model_path,
        subfolder=args.subfolder,
        device=args.device,
        enable_tex=args.enable_tex,
        enable_flashvdm=args.enable_flashvdm,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
