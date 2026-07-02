"""
Prompt templates for the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# ── RAG — answer strictly from retrieved context ──────────────────────────────

RAG_SYSTEM = (
    "You are a study assistant. Answer the question using ONLY the context nodes provided below — "
    "do not use your training data or general knowledge.\n\n"
    "HOW TO ANSWER:\n"
    "- If a node lists items or subtopics (e.g. 'RAG Types include: A, B, C'), those items ARE the answer — list them.\n"
    "- Summarise and present what the context says. Use bullet points when listing multiple items.\n"
    "- Cite sources using ONLY the exact node name from the context, written as [Node: <name>].\n"
    "  Never invent or guess a node name that is not shown in the context.\n"
    "- Do NOT add explanations, definitions, or details that are not present in the context.\n"
    "- If and ONLY if the context truly contains nothing relevant to the question, "
    "say: 'Your notes do not contain information about this topic.'\n\n"
    "Context from the user's notes:\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=RAG_SYSTEM),
        ("human", "{question}"),
    ]
)

# ── Web search — answer from search results ───────────────────────────────────

WEB_SYSTEM = (
    "You are a helpful study assistant. "
    "The user asked a question that goes beyond their personal study notes, "
    "so the following web search results have been retrieved to help you answer.\n\n"
    "Use the search results as your primary source. Be concise and accurate. "
    "Cite sources inline as (Source N) where appropriate.\n\n"
    "Search results:\n{search_results}"
)

web_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=WEB_SYSTEM),
        ("human", "{question}"),
    ]
)

# ── Intent classification ─────────────────────────────────────────────────────

INTENT_SYSTEM = (
    "You are a routing assistant. "
    "Classify whether the user's question is about their personal study notes / mind map "
    "or is a general knowledge / external question.\n"
    "Reply with exactly one word: NOTES or EXTERNAL."
)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=INTENT_SYSTEM),
        ("human", "{question}"),
    ]
)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string with node labels.

    Composite chunks (_comp) are marked with a GROUND TRUTH header so the LLM
    treats their item list as authoritative and does not substitute its own knowledge.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.get("metadata", {}).get("label") or chunk.get("doc_id", "")
        doc_id = chunk.get("doc_id", "")
        text = chunk["text"]
        if doc_id.endswith("_comp"):
            lines.append(
                f"[{i}] Node: {label} [AUTHORITATIVE LIST — use ONLY these items, no others]\n{text}"
            )
        else:
            lines.append(f"[{i}] Node: {label}\n{text}")
    return "\n\n".join(lines)


def build_web_context(results: list[dict]) -> str:
    """Format web search results into a numbered string."""
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", r.get("snippet", ""))
        href = r.get("href", r.get("url", ""))
        lines.append(f"Source {i}: {title}\n{body}\nURL: {href}")
    return "\n\n".join(lines)
