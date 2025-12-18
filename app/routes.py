import time
import requests
from flask import Blueprint, request, jsonify
from dependency_injector.wiring import inject, Provide
from containers import Container
from services.batik.rag import RAGService
import base64

api = Blueprint("api", __name__)

@api.route("/batik/chatbot", methods=["POST"])
@inject
def batik_chatbot(
    rag_service: RAGService = Provide[Container.rag_service],
):
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "query is required"}), 400

    result = rag_service.answer_query(query)

    return jsonify(result)

@api.route("/batik/generate-image", methods=["POST"])
@inject
def batik_generate_image(
    batik_generation_service = Provide[Container.batik_generation_service],
):
    image_bytes = batik_generation_service.generate()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    return jsonify({
        "generatedImage": f"data:image/png;base64,{b64}"
    })

@api.route("/batik/classify-image", methods=["POST"])
@inject
def batik_classify_image(
    batik_classification_service = Provide[Container.batik_classification_service],
):
    if 'image' not in request.files:
        return jsonify({"error": "image file is required"}), 400

    image_file = request.files['image']
    image_path = f"/tmp/{image_file.filename}"
    image_file.save(image_path)

    result = batik_classification_service.classify(image_path)

    return jsonify(result)

@api.route("/batik/generate-image-from-prompt", methods=["POST"])
@inject
def batik_generate_image_from_prompt(
    batik_image_service = Provide[Container.batik_image_service],
    generation_service = Provide[Container.batik_generation_service],
):
    data = request.get_json()
    prompt = data.get("prompt")
    steps = data.get("steps", 20)
    cfg_scale = data.get("cfg_scale", 7.5)
    
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    
    b64 = generation_service.generate_from_prompt(prompt, steps, cfg_scale)

    if not b64:
        return jsonify({"error": "Failed to generate image"}), 500
    
    image_path = f"/tmp/{prompt.replace(' ', '_')}_{int(time.time())}.png"

    image_bytes = base64.b64decode(b64)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    image_data = {
        "filename": f"{prompt.replace(' ', '_')}_{int(time.time())}.png",
        "path": image_path,
        "metadata": {
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
        }
    }
    
    batik_image_service.save_image(image_data)

    return jsonify({
        "generatedImage": f"data:image/png;base64,{b64}"
    })

@api.route("/batik/images", methods=["GET"])
@inject
def batik_get_all_images(
    batik_image_service = Provide[Container.batik_image_service],
):
    images = batik_image_service.get_all_images()
    for image in images:
        image['_id'] = str(image['_id'])
    return jsonify(images)