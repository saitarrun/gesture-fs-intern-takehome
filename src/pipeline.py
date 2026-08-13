"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from typing import Final, Protocol, TypedDict

from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


class LLMOutput(TypedDict):
    generated_text: str


class QuestionResult(TypedDict):
    answer: str
    sources: list[str]


class VectorStore(Protocol):
    def similarity_search(self, query: str, k: int = 3) -> list[Document]: ...


LLM = Callable[[str], list[LLMOutput]]
MODEL_NAME: Final[str] = "google/flan-t5-base"


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> LLM:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def generate(prompt: str) -> list[LLMOutput]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE: Final[str] = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store: VectorStore, llm: LLM, question: str) -> QuestionResult:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    docs: list[Document] = vector_store.similarity_search(question, k=3)
    if not docs:
        raise RuntimeError("No relevant sources were found.")

    sources: list[str] = [doc.page_content for doc in docs]
    context: str = "\n\n".join(sources)
    prompt: str = PROMPT_TEMPLATE.format(context=context, question=question)
    result: list[LLMOutput] = llm(prompt)
    answer: str = format_answer(result[0]["generated_text"])

    return {"answer": answer, "sources": sources}


def format_answer(answer: str) -> str:
    """Normalize generated text with capitalization and ending punctuation."""
    formatted: str = answer.strip()
    if not formatted:
        raise RuntimeError("The language model returned an empty answer.")

    if formatted[0].isalpha():
        formatted = formatted[0].upper() + formatted[1:]

    if not formatted.endswith((".", "!", "?")):
        formatted += "."

    return formatted


def print_result(result: QuestionResult) -> None:
    """Print retrieved sources followed by the generated answer."""
    print("\n📄 Sources:")
    for index, source in enumerate(result["sources"], start=1):
        print(f"  {index}. {source}")
    print(f"\n💬 Answer: {result['answer']}\n")


def validate_data_dir(data_dir: str) -> None:
    """Ensure the knowledge-base directory contains source documents."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not any(name.endswith(".txt") for name in os.listdir(data_dir)):
        raise FileNotFoundError(f"No .txt source files found in: {data_dir}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(argv: Sequence[str] | None = None) -> int:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    parser = argparse.ArgumentParser(description="Ask questions about the agency.")
    parser.add_argument(
        "--query",
        help="Answer one question and exit instead of starting the interactive CLI.",
    )
    args = parser.parse_args(argv)

    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")

    if args.query is not None and not args.query.strip():
        print("Error: Question cannot be empty.", file=sys.stderr)
        return 2

    try:
        validate_data_dir(data_dir)
        vector_store: VectorStore = build_knowledge_base(data_dir)
        llm: LLM = get_llm()
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.query is not None:
        print_result(ask_question(vector_store, llm, args.query))
        return 0

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if question.lower() == "quit":
            return 0
        if not question:
            print("Please enter a question, or type 'quit' to exit.")
            continue

        try:
            print_result(ask_question(vector_store, llm, question))
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
