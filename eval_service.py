import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Importaciones de Ragas y LangChain
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

# --- CONFIGURACIÓN DE AMBIENTE ---
# Estas variables se sincronizan con LangSmith automáticamente
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Evaluacion_RAG_Cloud")

app = FastAPI(title="RAG Evaluation Service")

# --- MODELOS DE DATOS (Pydantic) ---
class TestCase(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    project_name: str
    cases: List[TestCase]

# --- CONFIGURAR LLM CON GROQ ---
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    print("❌ ERROR: No se encontró GROQ_API_KEY en las variables de entorno")

# Inicializamos el evaluador con Llama 3.3 vía Groq
evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key
)
ragas_llm = LangchainLLMWrapper(evaluator_llm)

# --- RUTAS ---

@app.get("/")
def health_check():
    """Ruta necesaria para que Render sepa que el servicio está vivo."""
    return {
        "status": "ok", 
        "message": "RAG Eval Service is running",
        "project": os.environ.get("LANGCHAIN_PROJECT")
    }

@app.post("/evaluate-to-langsmith")
async def evaluate_to_langsmith(request: EvaluationRequest):
    try:
        # 1. Convertir los casos recibidos al formato que espera Ragas
        data_dicts = []
        for case in request.cases:
            data_dicts.append({
                "user_input": case.question,
                "response": case.answer,
                "retrieved_contexts": case.contexts,
                "reference": case.ground_truth
            })
        
        dataset = EvaluationDataset.from_list(data_dicts)
        
        # 2. Ejecutar la evaluación
        # Nota: Usamos Groq como LLM para que sea rápido y gratuito
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm
        )
        
        # 3. Enviar a LangSmith (se hace automáticamente por las variables de entorno)
        # Pero devolvemos los puntajes a n8n para que los veas
        return {
            "status": "success",
            "project_name": request.project_name,
            "scores": result.scores,
            "message": "Resultados enviados a LangSmith"
        }

    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- INICIO DEL SERVICIO (Configuración Render) ---
if __name__ == "__main__":
    # Render inyecta el puerto en la variable PORT. Por defecto usamos 10000.
    port = int(os.environ.get("PORT", 10000))
    # Importante: host 0.0.0.0 es obligatorio para Render
    uvicorn.run(app, host="0.0.0.0", port=port)
