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