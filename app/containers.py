from services.batik.classification import BatikClassificationService
from dependency_injector import containers, providers
from google import genai
from pinecone import Pinecone
import onnxruntime as ort

from services.batik.rag import RAGService
from services.batik.generator import BatikGenerationService
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

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