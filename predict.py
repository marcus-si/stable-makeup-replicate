# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
import inference_utils as inference 

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        snapshot_download(repo_id="msi2/Stable-Makeup-unofficial", repo_type="space", allow_patterns="checkpoints/*")
        self.pipeline, self.makeup_encoder = inference.init_pipeline()

    def predict(
        self,
        id_image_path: Path = Input(description="Photo of the person to apply makeup to"),
        makeup_image_path: Path = Input(description="Photo of the makeup style to apply"),
        guidance_scale: float = Input(
            description="1.05-1.15 is suggested for light makeup and 2 for heavy makeup", ge=1.01, le=3.0, default=1.6),
        size: int = Input(description="Size of the image to process", ge=256, le=1024, default=512),
    ) -> Path:
        return inference.inference(self.pipeline, self.makeup_encoder, id_image_path, makeup_image_path, guidance_scale, size)

        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
