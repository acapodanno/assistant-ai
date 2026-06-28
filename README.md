# 🌱 GreenThumb AI Assistant
### Assistente AI per il supporto clienti di un e-commerce

---

## 📋 Contesto aziendale

**GreenThumb Marketplace** è un e-commerce specializzato in articoli per il giardinaggio.
Riceve centinaia di richieste al giorno di natura eterogenea:

- 🌿 Domande sui prodotti
- 📦 Stato degli ordini
- 🪴 Consigli agronomici
- 🔄 Gestione resi

L'azienda introduce un assistente AI che risponde autonomamente alle richieste di **primo livello**, recupera informazioni dal catalogo e consulta lo stato degli ordini, mantenendo coerenza nella conversazione.

---

## 🎯 Obiettivi del progetto

| # | Obiettivo |
|---|-----------|
| 1 | Implementare un agente con loop **ReAct** usando **LangChain** e **LiteLLM** |
| 2 | Esporre almeno **3 tool**: ricerca catalogo (RAG), stato ordine, escalation operatore umano |
| 3 | Gestire **memoria a breve termine** (trimming o summarization) e **memoria a lungo termine** semantica (RAG su knowledge base aziendale) |
| 4 | Produrre risposte in **structured output** JSON: `answer`, `confidence`, `sources`, `needs_human` |
| 5 | Valutare l'agente con almeno una metrica **RAGAS** e una **DeepEval** |
| 6 | Deployare l'agente come API con **FastAPI** o **Chainlit** in locale |

---

## 📁 Struttura del progetto

```
assistant-ai/
│
├── 📄 .env                          # API keys (OPENAI_API_KEY, ecc.)
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 data/
│   ├── 📄 orders.json               # 20+ ordini fittizi
│   └── 📁 knowledge_base/
│       ├── 📁 prodotti/             # Schede prodotto (.txt/.pdf)
│       ├── 📁 guide/                # Guide agronomiche
│       └── 📁 policy/               # Policy resi e spedizioni
│
├── 📁 src/
│   ├── 📄 schemas.py                # Pydantic: AgentResponse
│   ├── 📁 agent/
│   │   ├── 📄 react_agent.py        # Loop ReAct (LangChain)
│   │   ├── 📄 tools.py              # 3 tool: catalog, ordine, escalation
│   │   └── 📄 memory.py             # Trimming + RAG memory
│   ├── 📁 rag/
│   │   ├── 📄 ingestion.py          # Chunking + embedding → ChromaDB
│   │   └── 📄 retriever.py          # Retriever semantico
│   └── 📁 api/
│       ├── 📄 main.py               # FastAPI app
│       └── 📄 chainlit_app.py       # UI chat locale
│
├── 📁 evaluation/
│   ├── 📄 test_dataset.json         # ≥15 conversazioni di test
│   ├── 📄 eval_ragas.py             # Metrica RAGAS (es. faithfulness)
│   └── 📄 eval_deepeval.py          # Metrica DeepEval (es. G-Eval)
│
└── 📁 tests/
    └── 📄 test_agent.py             # Pytest unit tests
```

---

## 📦 Dataset

- **Knowledge base**: almeno **30 documenti** tra schede prodotto, guide agronomiche e policy resi/spedizioni
- **`data/orders.json`**: almeno **20 ordini** fittizi con campi: id, cliente, prodotti, stato, data prevista

---

## 💬 Esempio di interazione

**Request:**

```http
POST /chat
Content-Type: application/json

{
  "message": "Quando arriva l'ordine 1042? Posso piantare adesso i bulbi di tulipano?"
}
```

**Response:**

```json
{
  "answer": "L'ordine 1042 arriva domani. I tulipani vanno piantati in autunno (ottobre-novembre): conservali in luogo fresco fino a settembre.",
  "confidence": "high",
  "sources": ["guides/bulbi_autunnali.md"],
  "needs_human": false
}
```

---

## ⚙️ Installazione

```bash
# 1. Clona il repository
git clone <repo-url>
cd assistant-ai

# 2. Installa pipenv (se non già presente)
pip install pipenv

# 3. Installa tutte le dipendenze e crea il virtualenv
pipenv install

# 4. Installa anche le dipendenze di sviluppo (test, linting)
pipenv install --dev

# 5. Attiva il virtualenv
pipenv shell

# 6. Configura le variabili d'ambiente
cp .env.example .env
# → Inserisci la tua OPENAI_API_KEY (o altra chiave LLM) nel file .env
```

> **Tip**: usa `pipenv run <comando>` per eseguire comandi senza attivare la shell:
> ```bash
> pipenv run uvicorn src.api.main:app --reload
> pipenv run pytest
> ```

---

## 🚀 Avvio

### FastAPI

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

### Chainlit

```bash
chainlit run src/api/chainlit_app.py
# → http://localhost:8000
```

---

## 🧪 Valutazione

```bash
# Metriche RAGAS
python evaluation/eval_ragas.py

# Metriche DeepEval
python evaluation/eval_deepeval.py
```

Il dataset di evaluation contiene **almeno 15 conversazioni** con domande attese, risposte di riferimento e commento ai risultati ottenuti.

---

## 📝 Convenzioni di codice

- **Nomi di variabili, funzioni**: `snake_case` sempre in inglese
- **Nomi di classi**: `CamelCase` sempre in inglese
- **Documentazione**: `docstring` obbligatoria per ogni funzione e classe
- **Formattazione**: `black` + `ruff`

---

## 🛠️ Stack tecnologico

| Componente | Tecnologia |
|---|---|
| LLM Framework | LangChain + LiteLLM |
| Vector Store | ChromaDB |
| Embedding | sentence-transformers |
| API | FastAPI + Uvicorn |
| UI Chat | Chainlit |
| Evaluation | RAGAS + DeepEval |
| Validazione | Pydantic v2 |
| Testing | Pytest |

---

## 📄 Licenza

Progetto a scopo didattico — GreenThumb Marketplace è un'azienda fittizia.
