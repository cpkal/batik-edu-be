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
    data = request.get_json()
    # seed = data.get("seed")

    image_bytes = batik_generation_service.generate()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    return jsonify({
        "generatedImage": f"data:image/png;base64,{b64}"
    })

    response = app.response_class(
        response=image_bytes,
        status=200,
        mimetype="image/png"
    )
    return response