import spaces
import gradio as gr
from inference_utils import inference


@spaces.GPU
def send_to_model(id_image, makeup_image, guidance_scale):
    if guidance_scale is None:
        # when creating example caches.
        guidance_scale = 1.6
    return inference(id_image, makeup_image, guidance_scale, size=512)

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML(
        """
            <h1 style="text-align: center; font-size: 32px; font-family: 'Times New Roman', Times, serif;">
                Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model
            </h1>
            <p style="text-align: center; font-size: 20px; font-family: 'Times New Roman', Times, serif;">
                <a style="text-align: center; display:inline-block"
                    href="https://xiaojiu-z.github.io/Stable-Makeup.github.io/">
                    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm.svg#center"
                    alt="Paper Page">
                </a>
                <a style="text-align: center; display:inline-block" href="https://huggingface.co/spaces/sky24h/Stable-Makeup-unofficial?duplicate=true">
                    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg#center" alt="Duplicate Space">
                </a>
            </p>
            """
    )
        gr.Interface(
            fn=send_to_model,
            inputs=[
                gr.Image(type="pil", label="id_image", height=512, width=512),
                gr.Image(type="pil", label="makeup_image", height=512, width=512),
                gr.Slider(minimum=1.01, maximum=3, value=1.6, step=0.05, label="guidance_scale", info="1.05-1.15 is suggested for light makeup and 2 for heavy makeup."),
            ],
            outputs="image",
            allow_flagging="never",
            description="This is an unofficial demo for the paper 'Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model'.",
            examples=[
                ["./test_imgs/id/1.jpg", "./test_imgs/makeup/1.jpg"],
                ["./test_imgs/id/2.jpg", "./test_imgs/makeup/2.jpg"],
                ["./test_imgs/id/3.jpg", "./test_imgs/makeup/3.jpg"],
                ["./test_imgs/id/4.jpg", "./test_imgs/makeup/4.png"],
            ],
            cache_examples=True,
        )
        demo.queue(max_size=10).launch()
