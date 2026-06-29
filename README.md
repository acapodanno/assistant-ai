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
pipenv install
pipenv shell
```

Configura le variabili d'ambiente copiando il file di esempio:
```bash
cp .env.example .env
```

Poi modifica `.env` con le tue credenziali (vedi sezione [Autenticazione GitHub](#-autenticazione-github-chainlit-oauth2) qui sotto):
```env
OPENAI_API_KEY=sk-...
PYTHONPATH=.

# GitHub OAuth2 (richiesto per l'interfaccia Chainlit)
OAUTH_GITHUB_CLIENT_ID=...
OAUTH_GITHUB_CLIENT_SECRET=...
CHAINLIT_AUTH_SECRET=...
```

---

## 🔐 Autenticazione GitHub (Chainlit OAuth2)

L'interfaccia Chat Chainlit usa **GitHub come provider OAuth2** per autenticare gli utenti. Segui questi passaggi per configurarla:

### Step 1 — Crea una GitHub OAuth App
1. Vai su [GitHub → Settings → Developer settings → OAuth Apps](https://github.com/settings/developers)
2. Clicca su **"New OAuth App"**
3. Compila i campi:
   - **Application name**: `GreenThumb AI Assistant`
   - **Homepage URL**: `http://localhost:8501`
   - **Authorization callback URL**: `http://localhost:8501/auth/oauth/github/callback`
4. Clicca **"Register application"**
5. Copia il **Client ID** e genera un **Client Secret**

### Step 2 — Genera il CHAINLIT_AUTH_SECRET
```bash
# Genera una chiave casuale sicura
pipenv run chainlint create-secret
```

### Step 3 — Aggiorna il file `.env`
```env
OAUTH_GITHUB_CLIENT_ID=<il tuo Client ID>
OAUTH_GITHUB_CLIENT_SECRET=<il tuo Client Secret>
CHAINLIT_AUTH_SECRET=<il valore generato al Step 2>
```

> **Nota:** Per un deploy in produzione, sostituisci `localhost:8501` con il tuo dominio reale nella callback URL dell'OAuth App GitHub.

---

## 🚀 Esecuzione dei Servizi Web

### API FastAPI
```bash
pipenv run uvicorn src.api.main:app --reload --port 8000
# http://localhost:8000/docs
```

### Interfaccia Chat Chainlit (con login GitHub)
```bash
pipenv run chainlit run src/api/chainlit_app.py --port 8501
# http://localhost:8501  →  login automatico con GitHub
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


---

## 📚 Reference Documentation

| Libreria | Descrizione | Docs |
|---|---|---|
| **FastAPI** | Framework API REST asincrono | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| **Chainlit** | UI chat per agenti AI con OAuth2 | [docs.chainlit.io](https://docs.chainlit.io/overview) |
| **LiteLLM** | Proxy unificato per LLM (OpenAI, Anthropic, ecc.) | [docs.litellm.ai](https://docs.litellm.ai/) |
| **LangChain** | Framework per catene RAG e retrieval | [docs.langchain.com](https://docs.langchain.com/) |
| **Ragas** | Valutazione automatica pipeline RAG | [docs.ragas.io](https://docs.ragas.io/en/stable/) |
| **DeepEval** | Framework di testing per LLM e agenti | [docs.deepeval.ai](https://docs.deepeval.ai/) |
| **Pydantic** | Validazione e serializzazione dei dati | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) |
| **Loguru** | Logging strutturato e leggibile | [pypi.org/loguru](https://pypi.org/project/loguru/) |

---

## 🔮 Next Implementation

- **MCP (Model Context Protocol)** — Integrazione con server MCP per esporre tool e risorse all'agente in modo standardizzato
- **Redis** — Sostituzione dell'in-memory store con Redis per la gestione persistente delle sessioni e della chat history tra riavvii del server
- **Database reale** — Connessione a PostgreSQL/MongoDB per ordini, ticket e profili utente al posto dei file JSON statici
- **Cloud Provider** — Deploy dell'intera stack su infrastruttura cloud:
  - **AWS**: FastAPI su ECS/Fargate, Chainlit su App Runner, Redis con ElastiCache, ChromaDB su EFS o migrazione a OpenSearch
  - **Azure**: Container Apps, Azure Cache for Redis, Azure OpenAI Service per i modelli LLM
