# 🌱 GreenThumb AI Assistant
### Assistente AI per il supporto clienti di un e-commerce

---

## 📋 Contesto aziendale

**GreenThumb Marketplace** è un e-commerce specializzato in articoli per il giardinaggio.
L'assistente AI risponde autonomamente alle richieste di primo livello, recupera informazioni dalla knowledge base (RAG) e consulta lo stato degli ordini, effettuando l'escalation con apertura ticket quando necessario.

---

## 📁 Struttura del progetto aggiornata

```
assistant-ai/
│
├── 📄 .env                          # Chiavi API (OpenAI)
├── 📄 Pipfile                       # Dipendenze Pipenv
├── 📄 main.py                       # [NUOVO] Script di test integrato 100% auticonsistente
├── 📄 GreenThumb_Assistant_Colab.ipynb # [NUOVO] Notebook Jupyter interattivo per Colab/Locale
├── 📄 README.md
│
├── 📁 data/
│   ├── 📄 orders.json               # Database degli ordini di test
│   └── 📁 knowledge_base/
│       ├── 📁 prodotti/             # Schede prodotto (.txt/.pdf)
│       ├── 📁 guide/                # Guide agronomiche
│       └── 📁 policy/               # Policy resi e spedizioni
│
├── 📁 src/
│   ├── 📁 agent/
│   │   ├── 📄 react_agent.py        # Agente autonomo ReAct (LiteLLM)
│   │   ├── 📄 tools.py              # Tool: get_order_by_id, create_ticket, rag_knowledge_base
│   │   └── 📁 models/               # Modelli Pydantic (ticket.py, agent_response.py)
│   ├── 📁 rag/
│   │   ├── 📄 ingestion.py          # Chunking + loading
│   │   ├── 📄 retriever.py          # Retriever ibrido (ChromaDB MMR + BM25 con fusione RRF)
│   │   ├── 📄 rag_chain.py          # Catena RAG LCEL (citazione obbligatoria [source.md])
│   │   └── 📄 run_ingestion.py      # Esecuzione indicizzazione database RAG
│   └── 📁 api/
│       ├── 📄 main.py               # FastAPI app
│       └── 📄 chainlit_app.py       # Interfaccia Chat Web Chainlit
│
└── 📁 evaluation/
    ├── 📄 test_dataset_rag.json     # [NUOVO] 15 domande KB pure per RAG
    ├── 📄 test_dataset_agent.json   # [NUOVO] 10 scenari complessi per l'Agente
    ├── 📄 eval_ragas.py             # Valutazione RAG pura (Ragas 0.4.x)
    └── 📄 eval_deepeval.py          # Valutazione Agent & Tools (DeepEval)
```

---

## ⚙️ Installazione ed Avvio Rapido

### 1. Setup Locale
```bash
# Installa dipendenze ed entra nel virtualenv
pipenv install --dev
pipenv shell

# Configura le chiavi API nel file .env
cp .env.example .env # ed imposta la tua OPENAI_API_KEY
```
## 🚀 Esecuzione dei Servizi Web

### API FastAPI
```bash
pipenv run uvicorn src.api.main:app --reload --port 8000
# http://localhost:8000/docs
```

### Interfaccia Chat Chainlit
```bash
pipenv run chainlit run src/api/chainlit_app.py --port 8501
# http://localhost:8501
```

---

## 🧪 Suite di Valutazione Disaccoppiata

Ho separato e ottimizzato la valutazione per testare i componenti singolarmente evitando degradazioni:

### A. Valutazione RAG (Ragas)
Valuta Faithfulness, Answer Relevancy e Context Recall sul dataset `evaluation/test_dataset_rag.json` caricando il retriever ibrido originale (BM25 + MMR).
```bash
pipenv run ./evaluation/eval_ragas.py
```

### B. Valutazione Agente & Strumenti (DeepEval)
Valuta l'Answer Relevancy e la Correctness (GEval) dell'agente nel gestire escalation, ticket, e lookup di ordini sul dataset `evaluation/test_dataset_agent.json`.
```bash
pipenv run ./evaluation/eval_deepeval.py
```
