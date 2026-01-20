import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Importaciones de Ragas y LangChain
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

# --- CONFIGURACIÓN DE AMBIENTE ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Evaluacion_RAG_Cloud")

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

# --- CONFIGURAR LLM Y EMBEDDINGS ---
groq_key = os.getenv("GROQ_API_KEY")

# Inicializamos el evaluador (Groq)
evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key
)
ragas_llm = LangchainLLMWrapper(evaluator_llm)

# Inicializamos Embeddings de HuggingFace (Para evitar el error de OpenAI)
# Usamos un modelo muy ligero para que Render no de Timeout
encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(encoder)

# --- RUTAS ---

@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "message": "RAG Eval Service is running",
        "project": os.environ.get("LANGCHAIN_PROJECT")
    }

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
        
        # Ejecutamos la evaluación pasando explícitamente el motor de embeddings
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        return {
            "status": "success",
            "project_name": request.project_name,
            "scores": result.scores,
            "message": "Resultados enviados a LangSmith"
        }

    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- INICIO DEL SERVICIO ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
