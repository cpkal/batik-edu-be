

from PIL import Image
import numpy as np
import onnxruntime as ort


class BatikClassificationService:
  def __init__(self, session):
      self.session = session

  def classify(self, filePath):
    input_tensor = self._preprocess_image(filePath)

    input_name = self.session.get_inputs()[0].name
    feeds = {input_name: input_tensor}

    results = self.session.run(None, feeds)
    outputs = results

    # softmax
    exp_outputs = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    softmax_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

    predicted_class = np.argmax(softmax_outputs, axis=1)[0]
    confidence = float(softmax_outputs[0][predicted_class])

    return {
      "predicted_class": int(predicted_class),
      "confidence": confidence
    }

  def _preprocess_image(self, filePath):
    img = Image.open(filePath).convert("RGB")
    img = img.resize((256, 256))  

    img_data = np.array(img).astype('float32') / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std

    # hwc to chw
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)  # add batch dimension

    tensor = img_data.astype(np.float32)

    return tensor
       