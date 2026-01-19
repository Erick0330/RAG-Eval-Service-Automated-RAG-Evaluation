# 🤖 RAG-Eval Service: Automated RAG Evaluation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b.svg)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq-orange.svg)](https://groq.com/)

Este servicio es un bridge de evaluación automatizada para pipelines de **RAG (Retrieval-Augmented Generation)**. Permite recibir casos de prueba desde herramientas como **n8n**, evaluarlos usando **Ragas** con modelos de **Groq** y visualizar los resultados detallados en **LangSmith**.

---

## 🚀 Características Principales

* **Métricas de Ragas:** Evaluación de *Faithfulness*, *Answer Relevancy*, *Answer Correctness* y *Context Precision*.
* **Groq Integration:** Inferencia ultra rápida usando Llama 3.3-70b sin costo de OpenAI.
* **Local Embeddings:** Uso de `sentence-transformers` ejecutados localmente para evitar costos de API.
* **LangSmith Native:** Trazabilidad completa de cada evaluación.
* **Cloud Ready:** Configurado para despliegue inmediato en **Render** o **Hugging Face**.

---

## 🛠️ Stack Tecnológico

* **Framework:** FastAPI (Python)
* **Evaluación:** Ragas
* **LLM:** Groq (Llama-3.3-70b-versatile)
* **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
* **Tracking:** LangSmith

---

## 📦 Instalación Local

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/rag-eval-service.git](https://github.com/tu-usuario/rag-eval-service.git)
    cd rag-eval-service
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar variables de entorno:**
    Crea un archivo `.env` o configúralas en tu panel de hosting:
    ```bash
    GROQ_API_KEY=tu_clave_de_groq
    LANGCHAIN_API_KEY=tu_clave_de_langsmith
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=RAG_Evaluation_N8N
    ```

4.  **Ejecutar el servidor:**
    ```bash
    python eval_service.py
    ```

---

## 📡 API Reference

### Evaluar Dataset
`POST /evaluate-to-langsmith`

**Request Body:**
```json
{
  "project_name": "Test_N8N",
  "cases": [
    {
      "question": "¿Cuál es el horario?",
      "answer": "El horario es de 9am a 5pm",
      "contexts": ["Nuestro horario de atención es de lunes a viernes de 9am a 5pm"],
      "ground_truth": "De 9am a 5pm"
    }
  ]
}
