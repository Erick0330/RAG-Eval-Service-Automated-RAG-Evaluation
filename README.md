---
title: RAG Precision & Evaluation Engine
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 RAG Evaluation Service (Ragas + Groq + LangSmith)

Este servicio proporciona un motor de evaluación automatizado para sistemas de **Generación Aumentada por Recuperación (RAG)**. Utiliza el framework **Ragas** y modelos de lenguaje de última generación (**Llama 3.3 70B vía Groq**) para auditar la calidad de las respuestas en base a cuatro pilares científicos.

## 🔬 Marco Teórico de Evaluación

El motor analiza la relación entre la **Pregunta**, los **Contextos Recuperados** y la **Respuesta Generada** mediante las siguientes métricas:

| Métrica | Dimensión | Descripción Científica |
| :--- | :--- | :--- |
| **Faithfulness** | Generación | Mide la consistencia factual de la respuesta con el contexto recuperado (evita alucinaciones). |
| **Answer Relevancy** | Generación | Evalúa qué tan directa y completa es la respuesta respecto a la consulta del usuario. |
| **Context Precision** | Recuperación | Califica la calidad del ranking de los documentos recuperados (S/N ratio). |
| **Context Recall** | Recuperación | Verifica si toda la información necesaria para responder fue efectivamente encontrada. |



## 🛠️ Stack Tecnológico

* **Motor de Evaluación:** [Ragas](https://docs.ragas.io/) (Retrieval-Augmented Generation Assessment).
* **Inferencia:** [Groq Cloud](https://groq.com/) (Llama 3.3 70B Versatile).
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` vía Hugging Face.
* **Observabilidad:** [LangSmith](https://smith.langchain.com/) para trazado de experimentos.
* **API:** FastAPI (Python 3.10+).

## 🚀 Guía de Uso (API Endpoint)

### `POST /evaluate-for-sheets`

Envía un batch de casos de prueba para obtener un análisis detallado compatible con Google Sheets o n8n.

**Cuerpo de la petición (JSON):**
```json
{
  "project_name": "GDS_Turismo_V2",
  "cases": [
    {
      "question": "¿Cómo accedo al módulo de autos?",
      "answer": "Debes ir a la pestaña superior...",
      "contexts": ["Manual Usuario pág 45: El módulo de autos se encuentra..."],
      "ground_truth": "El acceso se realiza mediante el menú superior, sección vehículos."
    }
  ]
}
