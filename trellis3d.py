import asyncio
import os
import uuid
import zipfile
import tempfile
from io import BytesIO
from PIL import Image as PILImage # Renamed to avoid conflict with chutes.Image
from fastapi import UploadFile, File, Form # For FastAPI specific type hints
import base64
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Callable, Union
from loguru import logger
import base64
import json
import time

from chutes.chute import Chute, ChutePack, NodeSelector
from chutes.image import Image # This is chutes.Image, not PIL.Image

# readme file
# read README.md file from local directory
readme = open("README_chutes.md", "r").read()


image = (
    Image(
    username="desudesuka",
    name="trellis",
    tag=f"{time.strftime('%Y-%m-%d')}-{int(time.time())}",
    readme=readme,
    )
    .from_base("parachutes/base-python:3.10.17")
    .run_command("pip install xformers==0.0.27.post2 torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121")
    .run_command("git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /tmp/trellis")
    .run_command("pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers")
    .run_command("pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")
    .run_command("mkdir -p /tmp/extensions && cd /tmp/extensions &&  git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast &&  pip install /tmp/extensions/nvdiffrast")
    .run_command("mkdir -p /tmp/extensions &&  git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0' pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/")
    .run_command("pip install spconv-cu124")
    .run_command("pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html")
    .run_command("pip install xformers==0.0.27.post2 torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121")
)

class TrellisRequest(BaseModel):
    seed: Optional[int] = Field(1, description="Random seed for generation.")
    simplify: Optional[float] = Field(
        0.95, ge=0.0, le=1.0, description="Ratio of triangles to remove in GLB simplification."
    )
    texture_size: Optional[int] = Field(
        1024, ge=256, le=4096, description="Size of the texture for GLB export."
    )
    image_b64: str
    gaussian: Optional[bool] = Field(
        True, description="Whether to generate Gaussian Splat output."
    )
    radiance_field: Optional[bool] = Field(
        True, description="Whether to generate Radiance Field output."
    )
    mesh: Optional[bool] = Field(
        True, description="Whether to generate Mesh output."
    )


# --- Build Function ---

predict = Chute(
    username="desudesuka",
    name="trellis",
    tagline="3D Generation with Trellis",
    readme=readme,
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    )
)

@predict.on_startup()
async def initialize(self):
    """
    Load the classification pipeline and perform a warmup.
    """
    logger.info("Setting environment variables for Trellis...")
    # These might be better in the Dockerfile ENV,
    # but setting here ensures they are active for this process.
    # Ensure these os.environ calls are done before importing trellis if it reads them at import time.
    os.environ['ATTN_BACKEND'] = os.getenv('ATTN_BACKEND', 'xformers')
    os.environ['SPCONV_ALGO'] = os.getenv('SPCONV_ALGO', 'native')
    logger.info(f"ATTN_BACKEND set to: {os.environ['ATTN_BACKEND']}")
    logger.info(f"SPCONV_ALGO set to: {os.environ['SPCONV_ALGO']}")

    logger.info("Importing Trellis and related libraries...")
    # Import heavy libraries here to keep initial Chute load faster if possible
    # and to ensure ENV vars are set before import.
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    import imageio
   
    logger.info("Loading Trellis pipeline...")
    # Load a pipeline from a model folder or a Hugging Face model hub.
    self.pipeline = TrellisImageTo3DPipeline.from_pretrained("jetx/trellis-image-large")
    logger.info("Moving pipeline to CUDA...")
    self.pipeline.cuda()
    logger.info("Trellis pipeline initialized and moved to CUDA.")

    self.render_utils = render_utils
    self.postprocessing_utils = postprocessing_utils
    self.imageio = imageio

    # Warmup the pipeline with a dummy image
    dummy_image = PILImage.new("RGB", (512, 512), color=(255, 255, 255))
    logger.info("Warming up the pipeline with a dummy image...")
    self.pipeline.run(
        dummy_image,
        seed=1,
        formats=["gaussian", "radiance_field", "mesh"],
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    logger.info("Pipeline warmup completed.")



@predict.cord(
    public_api_path="/generate",
    method="POST",
    # Input schema is for the form fields, file is separate
    # For FastAPI, we define UploadFile directly in the function signature.
    # Pydantic model can be used for other form fields if sent as JSON payload with multipart,
    # or use Form fields. Here, we'll assume parameters come as form data.
    output_content_type="application/zip",
    pass_chute=True, # To access self.pipeline etc.
    input_schema=TrellisRequest,
    stream=False,
)
async def generate(
    self,
    params: TrellisRequest,
    # For TrellisRequest Pydantic model:
    # params: TrellisRequest (if sending JSON payload + file)
) -> JSONResponse:
    
    seed = params.seed
    formats = []
    if params.gaussian:
        formats.append("gaussian")
    if params.radiance_field:
        formats.append("radiance_field")
    if params.mesh:
        formats.append("mesh")

    simplify = params.simplify
    texture_size = params.texture_size
    image_b64 = params.image_b64
    params_dict = {
        "seed": seed,
        "formats": formats,
        "simplify": simplify,
        "texture_size": texture_size,
    }
    logger.info(f"Received generation request with params: {params_dict}")
                  
    image_bytes = base64.b64decode(image_b64)
    input_image = PILImage.open(BytesIO(image_bytes))

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Processing image in temporary directory: {temp_dir}")
        output_files_generated = []

        try:
            # Run the pipeline
            logger.info("Running Trellis pipeline...")
            # Construct optional params if you decide to add them later:
            # optional_params = {}
            # if params.sparse_structure_sampler_steps:
            #     optional_params["sparse_structure_sampler_params"] = {
            #         "steps": params.sparse_structure_sampler_steps,
            #         "cfg_strength": params.sparse_structure_sampler_cfg_strength,
            #     } # etc.

            pipeline_outputs = self.pipeline.run(
                input_image,
                seed=seed,
                formats=formats,
                # **optional_params
            )
            logger.info("Pipeline run completed. Processing outputs...")

            # Render and save outputs
            base_filename = str(uuid.uuid4())

            if "gaussian" in pipeline_outputs and pipeline_outputs["gaussian"]:
                gs = pipeline_outputs['gaussian'][0]
                video_gs = self.render_utils.render_video(gs)['color']
                gs_video_path = os.path.join(temp_dir, f"{base_filename}_gs.mp4")
                self.imageio.mimsave(gs_video_path, video_gs, fps=30)
                output_files_generated.append(gs_video_path)
                logger.info(f"Generated Gaussian Splat video: {gs_video_path}")

                ply_path = os.path.join(temp_dir, f"{base_filename}.ply")
                gs.save_ply(ply_path)
                output_files_generated.append(ply_path)
                logger.info(f"Saved Gaussian Splat PLY: {ply_path}")


            if "radiance_field" in pipeline_outputs and pipeline_outputs["radiance_field"]:
                rf = pipeline_outputs['radiance_field'][0]
                video_rf = self.render_utils.render_video(rf)['color']
                rf_video_path = os.path.join(temp_dir, f"{base_filename}_rf.mp4")
                self.imageio.mimsave(rf_video_path, video_rf, fps=30)
                output_files_generated.append(rf_video_path)
                logger.info(f"Generated Radiance Field video: {rf_video_path}")

            if "mesh" in pipeline_outputs and pipeline_outputs["mesh"]:
                mesh = pipeline_outputs['mesh'][0]
                video_mesh = self.render_utils.render_video(mesh)['normal']
                mesh_video_path = os.path.join(temp_dir, f"{base_filename}_mesh.mp4")
                self.imageio.mimsave(mesh_video_path, video_mesh, fps=30)
                output_files_generated.append(mesh_video_path)
                logger.info(f"Generated Mesh normal video: {mesh_video_path}")

                # GLB generation requires gaussian and mesh
                if "gaussian" in pipeline_outputs and pipeline_outputs["gaussian"]:
                    gs_for_glb = pipeline_outputs['gaussian'][0] # Re-access if needed
                    glb = self.postprocessing_utils.to_glb(
                        gs_for_glb,
                        mesh,
                        simplify=params_dict["simplify"],
                        texture_size=params_dict["texture_size"],
                    )
                    glb_path = os.path.join(temp_dir, f"{base_filename}.glb")
                    glb.export(glb_path)
                    output_files_generated.append(glb_path)
                    logger.info(f"Exported GLB file: {glb_path}")

            if not output_files_generated:
                logger.warning("No output files were generated based on formats or pipeline results.")
                raise Exception("No valid output formats were specified or generated.")

            # Create a ZIP file in memory
            zip_io = BytesIO()
            with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
                for f_path in output_files_generated:
                    temp_zip.write(f_path, arcname=os.path.basename(f_path))
            zip_io.seek(0)
            logger.info(f"Successfully zipped {len(output_files_generated)} files.")

            zip_bytes = zip_io.getvalue()
            zip_b64 = base64.b64encode(zip_bytes).decode('utf-8')

            return JSONResponse(
                content={
                    "content_type": "application/zip",
                    "status": "success",
                    "bytes": zip_b64,
                    "filename": f"{base_filename}_outputs.zip"
                },
    )

        except Exception as e:
            logger.exception("Error during Trellis pipeline execution or file processing:")
            # Return error as JSON for client to parse
            error_payload = json.dumps({"error": "Processing error", "detail": str(e)})
            raise Exception(
                status_code=500,
                content=iter([error_payload.encode('utf-8')]),
                media_type="application/json"
            )
