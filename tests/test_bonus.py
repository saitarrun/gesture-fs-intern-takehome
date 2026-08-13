"""Fast unit tests for bonus behavior and pipeline wiring."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.pipeline import ask_question, format_answer, main, validate_data_dir


def test_ask_question_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        ask_question(Mock(), Mock(), "   ")


def test_ask_question_uses_three_sources_and_formats_prompt():
    store = Mock()
    store.similarity_search.return_value = [
        Document(page_content="one"),
        Document(page_content="two"),
        Document(page_content="three"),
    ]
    llm = Mock(return_value=[{"generated_text": "answer"}])

    result = ask_question(store, llm, " question ")

    store.similarity_search.assert_called_once_with("question", k=3)
    prompt = llm.call_args.args[0]
    assert "one\n\ntwo\n\nthree" in prompt
    assert "Client question: question" in prompt
    assert result == {"answer": "Answer.", "sources": ["one", "two", "three"]}


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("yes", "Yes."),
        (
            "  the Growth package costs $5,500/month  ",
            "The Growth package costs $5,500/month.",
        ),
        ("already complete!", "Already complete!"),
        ("$5,500/month", "$5,500/month."),
    ],
)
def test_format_answer_adds_capitalization_and_punctuation(generated, expected):
    assert format_answer(generated) == expected


def test_validate_data_dir_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        validate_data_dir(str(tmp_path / "missing"))


@patch("src.pipeline.get_llm")
@patch("src.pipeline.build_knowledge_base")
def test_query_mode_answers_once(build_knowledge_base, get_llm, capsys):
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="pricing source")]
    build_knowledge_base.return_value = store
    get_llm.return_value = Mock(return_value=[{"generated_text": "$5,500/month"}])

    assert main(["--query", "Growth price?"]) == 0
    output = capsys.readouterr().out
    assert "pricing source" in output
    assert "$5,500/month" in output
    store.similarity_search.assert_called_once_with("Growth price?", k=3)


@patch("src.pipeline.get_llm", return_value=Mock())
@patch("src.pipeline.build_knowledge_base", return_value=Mock())
def test_interactive_mode_skips_empty_input(build_knowledge_base, get_llm, capsys):
    with patch("builtins.input", side_effect=["", "quit"]):
        assert main([]) == 0

    assert "Please enter a question" in capsys.readouterr().out
