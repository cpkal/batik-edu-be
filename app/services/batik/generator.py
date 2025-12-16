import onnxruntime as ort
import numpy as np
from PIL import Image
import io

class BatikGenerationService:
    def __init__(self, session: ort.InferenceSession, z_dim: int = 100):
      self.session = session
      self.z_dim = z_dim
      self.input_name = self.session.get_inputs()[0].name
      self.output_name = self.session.get_outputs()[0].name

    def generate_noise(self, dim: int = 100):
      # Uniform(-1, 1), shape [1, dim, 1, 1]
      z = np.random.uniform(-1, 1, size=(1, dim, 1, 1)).astype(np.float32)
      return z

    def generate(self) -> bytes:
      z = self.generate_noise()

      outputs = self.session.run(
          [self.output_name],
          {self.input_name: z}
      )

      output_tensor = outputs[0]
      return self._to_png(output_tensor)

    def _to_png(self, tensor: np.ndarray) -> bytes:
      """
      Handles output shape:
      - [1, C, H, W]
      - [C, H, W]
      """

      if tensor.ndim == 4:
          tensor = tensor[0]  # remove batch dim

      # tensor: [C, H, W]
      C, H, W = tensor.shape

      # CHW -> HWC
      img = tensor.transpose(1, 2, 0)

      # tanh [-1,1] -> uint8 [0,255]
      img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

      pil_img = Image.fromarray(img)
      buf = io.BytesIO()
      pil_img.save(buf, format="PNG")

      return buf.getvalue()