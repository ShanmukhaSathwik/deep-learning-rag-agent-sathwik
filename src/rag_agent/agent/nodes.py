"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    settings = get_settings()

    original_query = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            original_query = message.content
            break

    rewritten = original_query
    try:
        llm = LLMFactory(settings).create()
        rewrite_prompt = [
            SystemMessage(
                content=(
                    "Rewrite the user's question into a short keyword-style query "
                    "optimised for vector similarity search over deep learning notes. "
                    "Return only the rewritten query, with no explanation."
                )
            ),
            HumanMessage(content=original_query),
        ]
        result = llm.invoke(rewrite_prompt)
        if result.content and result.content.strip():
            rewritten = result.content.strip()
    except Exception:
        rewritten = original_query

    return {"original_query": original_query, "rewritten_query": rewritten}

# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = VectorStoreManager()

    chunks = manager.query(
        query_text=state.get("rewritten_query") or state.get("original_query", ""),
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter"),
    )

    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}

    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.get("no_context_found"):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])

    context_parts = []
    sources = []
    for chunk in chunks:
        context_parts.append(
            f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]\n{chunk.chunk_text}\n"
        )
        citation = chunk.to_citation()
        if citation not in sources:
            sources.append(citation)

    context_text = "\n".join(context_parts)
    confidence = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

    history = trim_messages(
        state["messages"],
        max_tokens=settings.max_context_tokens,
        token_counter=len,
        strategy="last",
        include_system=False,
        allow_partial=False,
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Retrieved context:\n\n{context_text}"),
        *history,
        HumanMessage(content=state.get("original_query", "")),
    ]

    result = llm.invoke(prompt_messages)
    answer = result.content

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", ""),
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    
    return "generate"