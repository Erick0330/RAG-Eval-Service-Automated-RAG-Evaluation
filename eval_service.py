import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Esquemas de datos (Rápido)
class EvaluationCase(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    project_name: str
    cases: List[EvaluationCase]

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Servidor activo"}

@app.post("/evaluate-to-langsmith")
async def evaluate_to_langsmith(request: EvaluationRequest):
    try:
        # IMPORTACIONES DENTRO DE LA FUNCIÓN (Lazy Loading)
        # Esto hace que el servidor arranque rápido y Render no dé error de puerto
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness, ContextPrecision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings

        # Configuración de ambiente
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RAG_Eval")

        # Configurar LLM y Embeddings
        evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        metrics = [Faithfulness(), AnswerRelevancy(), AnswerCorrectness(), ContextPrecision()]

        # Preparar datos
        data = {
            "question": [c.question for c in request.cases],
            "answer": [c.answer for c in request.cases],
            "contexts": [c.contexts for c in request.cases],
            "ground_truth": [c.ground_truth for c in request.cases]
        }
        dataset = Dataset.from_dict(data)

        # Evaluar
        result = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)
        
        return {"status": "success", "scores": result}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
