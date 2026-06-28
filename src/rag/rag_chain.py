from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = (
    "Sei l'assistente clienti di GreenThumb Marketplace, specializzato in giardinaggio e prodotti per l'orto. "
    "Rispondi alle domande dei clienti utilizzando ESCLUSIVAMENTE il contesto fornito dalla knowledge base.\n\n"
    "REGOLE:\n"
    "1. Cita sempre il nome del documento sorgente tra parentesi quadre (es. [spedizioni.md]).\n"
    "2. Se la risposta non è presente nel contesto, rispondi: "
    "'Non ho trovato questa informazione nella nostra knowledge base. "
    "Ti consiglio di contattare il supporto a supporto@greenthumb.it'\n"
    "3. Sii conciso, chiaro e usa un tono cordiale e professionale.\n"
    "4. Per le domande su prodotti, includi sempre il prezzo se disponibile nel contesto.\n\n"
    "CONTESTO:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "fonte sconosciuta")
        filename = source.split("/")[-1]
        parts.append(f"[{filename}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)

def setup_rag_chain(retriever=None):
    if retriever is not None:
        chain = (
            {"context": retriever | _format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    else:
        chain = prompt | llm | StrOutputParser()

        def invoke_with_docs(inputs: dict) -> dict:
            docs = inputs.get("context", [])
            context_str = _format_docs(docs)
            answer = chain.invoke({"input": inputs["input"], "context": context_str})
            return {"answer": answer, "context": docs}

        return invoke_with_docs
