"""
RAG chatbot for PaperMind.

Pipeline
--------
User question
    → history-aware retriever  (rephrases follow-up questions into
                                 standalone queries so retrieval is accurate)
    → vector store retrieval   (MMR / similarity / hybrid)
    → stuff-documents chain    (retrieved chunks injected into system prompt)
    → OpenAI GPT          (streamed response with source citations)
"""
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config



# Step 1 – rephrase follow-up questions into standalone queries
_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, which may "
        "reference context in the chat history, formulate a standalone "
        "question that can be understood without the chat history. "
        "Do NOT answer the question—just reformulate it if needed, "
        "otherwise return it as-is.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 2 – answer using retrieved paper chunks
_QA_SYSTEM_PROMPT = """\
You are **PaperMind**, an intelligent research assistant specialising in \
Generative AI and Large Language Model (LLM) research papers. \
You have access to the following curated papers:

  • "Attention Is All You Need" – the foundational Transformer paper
  • GPT-4 Technical Report
  • InstructGPT – reinforcement learning from human feedback (RLHF)
  • Gemini: A Family of Highly Capable Multimodal Models
  • Mistral 7B

**Your task:** Answer the user's question using *only* the context below. \
When you draw on a specific paper or passage, mention its title explicitly. \
If the context is insufficient to answer fully, say so honestly and suggest \
what additional information would help.

**Context from research papers:**
{context}

**Guidelines:**
- Ground every claim in the provided context; do not hallucinate.
- Cite which paper(s) and, where possible, which section or concept you are referencing.
- Use precise technical language suitable for AI/ML practitioners while remaining clear.
- When comparing approaches across papers, highlight key similarities and differences.
- If a question falls completely outside the scope of these papers, say so politely.
"""

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# LLM factory

def _get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        streaming=streaming,
    )


# Chain factory

def build_rag_chain(retriever: BaseRetriever):
    """
    Compose the full RAG chain:

        history-aware retriever  →  stuff-documents QA chain
    """
    llm = _get_llm()

    # Wrap retriever so follow-up questions are rephrased first
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, _CONTEXTUALIZE_PROMPT
    )

    # Chain that stuffs retrieved docs into the QA prompt
    qa_chain = create_stuff_documents_chain(llm, _QA_PROMPT)

    # Combine: retrieval + generation
    return create_retrieval_chain(history_aware_retriever, qa_chain)


# Chatbot class

class PaperMindChatbot:
    """
    Stateful chatbot that maintains conversation history across turns.

    Usage::

        chatbot = PaperMindChatbot(retriever)
        answer, sources = chatbot.chat("What is multi-head attention?")
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        self._chain = build_rag_chain(retriever)
        self.chat_history: List = []

    def chat(self, question: str) -> Tuple[str, List[Document]]:
        """
        Ask a question and return *(answer, source_documents)*.

        Chat history is updated automatically so follow-up questions work.
        """
        response = self._chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })

        answer: str = response["answer"]
        sources: List[Document] = response.get("context", [])

        # Append to history (LangChain message objects)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, sources

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.chat_history = []
