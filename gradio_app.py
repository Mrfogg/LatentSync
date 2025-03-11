import gradio as gr
from pathlib import Path
from scripts.inference_tmp import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import os

CONFIG_PATH = Path("configs/unet/second_stage_prod.yaml")


def get_model_files(model_dir):
    """Get all model files from the specified directory."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []

    model_files = []
    for ext in ['.pt', '.pth', '.ckpt']:
        model_files.extend([f for f in model_dir.glob(f'*{ext}')])
    return [str(f) for f in model_files]


def process_video(
        video_path,
        audio_path,
        model_path,
        guidance_scale,
        inference_steps,
        seed,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()
    model_path = Path(model_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed, model_path)
    args.gpu_id = 2
    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
        video_path: str,
        audio_path: str,
        output_path: str,
        inference_steps: int,
        guidance_scale: float,
        seed: int,
        model_path: str
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            model_path,
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


def create_interface(model_dir):
    # Get available model files
    model_files = get_model_files(model_dir)
    if not model_files:
        print(f"Warning: No model files found in {model_dir}")
        model_files = ["No models available"]

    # Create Gradio interface
    with gr.Blocks(title="LatentSync Video Processing") as demo:
        gr.Markdown(
            """
        # LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync
        Upload a video and audio file to process with LatentSync model.

        <div align="center">
            <strong>Chunyu Li1,2  Chao Zhang1  Weikai Xu1  Jinghui Xie1,†  Weiguo Feng1
            Bingyue Peng1  Weiwei Xing2,†</strong>
        </div>

        <div align="center">
            <strong>1ByteDance   2Beijing Jiaotong University</strong>
        </div>

        <div style="display:flex;justify-content:center;column-gap:4px;">
            <a href="https://github.com/bytedance/LatentSync">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://arxiv.org/pdf/2412.09262">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
        </div>
        """
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                audio_input = gr.Audio(label="Input Audio", type="filepath")
                model_input = gr.Dropdown(
                    choices=model_files,
                    label="Select UNet Model",
                    value=model_files[0] if model_files else None
                )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=1.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")

                with gr.Row():
                    seed = gr.Number(value=1247, label="Random Seed", precision=0)

                process_btn = gr.Button("Process Video")

            with gr.Column():
                video_output = gr.Video(label="Output Video")

                gr.Examples(
                    examples=[
                        ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                        ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                        ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                    ],
                    inputs=[video_input, audio_input],
                )

        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                audio_input,
                model_input,
                guidance_scale,
                inference_steps,
                seed,
            ],
            outputs=video_output,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing UNet model files")
    parser.add_argument("--port", type=int, default=7861, help="Port to run the server on")
    args = parser.parse_args()

    demo = create_interface(args.model_dir)
    demo.launch(inbrowser=True, share=False, server_port=args.port, server_name='0.0.0.0')
