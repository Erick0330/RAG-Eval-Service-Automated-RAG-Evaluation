import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from datasets import Dataset

# Importaciones de Ragas
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Importaciones de LangChain modernas
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIGURACIÓN DE AMBIENTE
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Evaluacion_RAG_N8N"

app = FastAPI()

# 1. Configurar LLM con Groq
evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="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"
)
ragas_llm = LangchainLLMWrapper(evaluator_llm)

# 2. Configurar Embeddings locales inicializados
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# 3. Inicializar métricas como OBJETOS (Esto corrige tu error 500)
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    AnswerCorrectness(),
    ContextPrecision()
]

class EvaluationCase(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    project_name: str
    cases: List[EvaluationCase]

@app.post("/evaluate-to-langsmith")
async def evaluate_to_langsmith(request: EvaluationRequest):
    try:
        data = {
            "question": [c.question for c in request.cases],
            "answer": [c.answer for c in request.cases],
            "contexts": [c.contexts for c in request.cases],
            "ground_truth": [c.ground_truth for c in request.cases]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Ejecutar evaluación con métricas inicializadas
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        return {
            "status": "success",
            "scores": result,
            "message": "Evaluación completada"
        }
        
    except Exception as e:
        print(f"ERROR DETECTADO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)