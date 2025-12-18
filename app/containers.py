from repositories.batik_image import BatikImageRepository
from services.batik.batik_image import BatikImageService
from services.batik.classification import BatikClassificationService
from dependency_injector import containers, providers
from google import genai
from pinecone import Pinecone
import onnxruntime as ort
from pymongo import MongoClient

from services.batik.rag import RAGService
from services.batik.generator import BatikGenerationService
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, MONGO_URI

class Container(containers.DeclarativeContainer):
  wiring_config = containers.WiringConfiguration(
    modules=["routes"]
  )

  gemini_client = providers.Singleton(
    genai.Client,
    api_key=GEMINI_API_KEY
  )

  pinecone_client = providers.Singleton(
    Pinecone,
    api_key=PINECONE_API_KEY,
  )

  pinecone_index = providers.Singleton(
    lambda pc: pc.Index(PINECONE_INDEX_NAME),
    pc=pinecone_client
  )

  rag_service = providers.Singleton(
    RAGService,
    index=pinecone_index,
    gemini_client=gemini_client
  )

  onnx_session = providers.Singleton(
    ort.InferenceSession,
    "models/generator_final.onnx",
    providers.Singleton(ort.SessionOptions)
  )

  batik_generation_service = providers.Singleton(
    lambda session, z_dim: BatikGenerationService(session, z_dim),
    session=onnx_session,
    z_dim=100
  )

  onnx_session_classification = providers.Singleton(
    ort.InferenceSession,
    "models/mobilenetv3_batik_best_v2.onnx",
    providers.Singleton(ort.SessionOptions)
  )

  batik_classification_service = providers.Singleton(
    lambda session: BatikClassificationService(session),
    session=onnx_session_classification
  )

  # mongo using mongodb atlas
  mongo_client = providers.Singleton(
    MongoClient,
    MONGO_URI
  )

  mongo_db = providers.Singleton(
    lambda client: client.get_database("batik_edu_db"),
    client=mongo_client
  )

  batik_collection = providers.Factory(
    lambda db: db.get_collection("batik_images"),
    db=mongo_db
  )

  batik_image_repository = providers.Factory(
    BatikImageRepository,
    collection=batik_collection
  )

  batik_image_service = providers.Singleton(
    lambda repository: BatikImageService(repository),
    repository=batik_image_repository
  )