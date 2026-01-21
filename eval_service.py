import os
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import traceback

# Importaciones de Ragas y LangChain
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

app = FastAPI(title="RAG Evaluation to Sheets Service")

# --- CONFIGURAR MODELOS ---
groq_key = os.getenv("GROQ_API_KEY")
base_model = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=groq_key,
    n=1, 
    temperature=0
)

ragas_llm = LangchainLLMWrapper(base_model)
encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(encoder)

# --- MODELOS DE DATOS ---
class TestCase(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    project_name: str
    cases: List[TestCase]

# --- UTILIDAD PARA LIMPIAR DATOS ---
def limpiar_valor(val):
    """Asegura que los valores sean compatibles con JSON y Sheets (evita NaN/Inf)"""
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return float(val)

@app.post("/evaluate-for-sheets")
async def evaluate_for_sheets(request: EvaluationRequest):
    try:
        # 1. Preparar el dataset para Ragas
        data_dicts = []
        for case in request.cases:
            data_dicts.append({
                "user_input": case.question,
                "response": case.answer,
                "retrieved_contexts": case.contexts,
                "reference": case.ground_truth
            })
        
        dataset_ragas = EvaluationDataset.from_list(data_dicts)
        
        # 2. Ejecutar Evaluación
        result = evaluate(
            dataset=dataset_ragas,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        # 3. Convertir a Pandas para facilitar la iteración
        df = result.to_pandas()
        
        # 4. Construir el arreglo de objetos para el cliente
        filas_para_sheet = []
        for _, row in df.iterrows():
            objeto_fila = {
                "pregunta": row["user_input"],
                # Métricas redondeadas a 4 decimales
                "fidelidad": limpiar_valor(row["faithfulness"]),
                "relevancia": limpiar_valor(row["answer_relevancy"]),
                "precision_contexto": limpiar_valor(row["context_precision"]),
                "recuperacion_contexto": limpiar_valor(row["context_recall"])
            }
            filas_para_sheet.append(objeto_fila)

        # 5. Devolver el arreglo listo para Sheets
        return {
            "status": "success",
            "project": request.project_name,
            "data": filas_para_sheet
        }

    except Exception as e:
        print(f"❌ Error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)