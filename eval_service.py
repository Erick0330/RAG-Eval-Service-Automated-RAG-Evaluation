import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Importaciones de Ragas y LangChain
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

# --- CONFIGURACIÓN DE AMBIENTE ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Evaluacion_RAG_Cloud")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="RAG Evaluation Service")

# --- MODELOS DE DATOS ---
class TestCase(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    project_name: str
    cases: List[TestCase]

# --- CONFIGURAR LLM (GROQ) Y EMBEDDINGS (OPENAI) ---
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not groq_key or not openai_key:
    print("❌ ERROR: Faltan llaves de API (GROQ o OPENAI) en las variables de entorno")

# 1. Evaluador principal (Groq - Llama 3.3)
evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
ragas_llm = LangchainLLMWrapper(evaluator_llm)

# 2. Embeddings (OpenAI - Muy ligero y rápido para Render)
encoder = OpenAIEmbeddings(api_key=openai_key)
ragas_embeddings = LangchainEmbeddingsWrapper(encoder)

# --- RUTAS ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Eval Service is running with Groq + OpenAI Embeddings"}

@app.post("/evaluate-to-langsmith")
async def evaluate_to_langsmith(request: EvaluationRequest):
    try:
        data_dicts = []
        for case in request.cases:
            data_dicts.append({
                "user_input": case.question,
                "response": case.answer,
                "retrieved_contexts": case.contexts,
                "reference": case.ground_truth
            })
        
        dataset = EvaluationDataset.from_list(data_dicts)
        
        # Ejecutar evaluación
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        return {
            "status": "success",
            "scores": result.scores,
            "message": "Enviado a LangSmith con éxito"
        }

    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
