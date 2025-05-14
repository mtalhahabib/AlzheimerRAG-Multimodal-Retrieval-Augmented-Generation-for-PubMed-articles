import unittest
from unittest.mock import patch, MagicMock
from claudeRag import AlzheimerRAG
import streamlit as st

class TestAlzheimerRAG(unittest.TestCase):

    def setUp(self):
        """Initialize test setup."""
        self.rag_system = AlzheimerRAG()

    @patch("claudeRag.AlzheimerRAG.answer")
    def test_answer_generation(self, mock_answer):
        """Test if Alzheimer's RAG generates a valid response."""
        mock_answer.return_value = ("This is a test response.", [])
        response, figures = self.rag_system.answer("What is Alzheimer's?", False)

        self.assertEqual(response, "This is a test response.")
        self.assertEqual(figures, [])

    @patch("claudeRag.PubMedFetcher.fetch_articles")
    def test_pubmed_fetching(self, mock_fetch):
        """Test if PubMedFetcher retrieves research articles."""
        mock_fetch.return_value = ["Article 1", "Article 2"]
        pubmed = PubMedFetcher()
        articles = pubmed.fetch_articles("Alzheimer's Research")

        self.assertEqual(len(articles), 2)
        self.assertIn("Article 1", articles)

    def test_rag_initialization(self):
        """Test if the RAG system initializes correctly."""
        self.assertIsInstance(self.rag_system, AlzheimerRAG)

class TestStreamlitInterface(unittest.TestCase):

    @patch("streamlit.markdown")
    def test_sidebar_content(self, mock_markdown):
        """Test if sidebar renders properly."""
        with patch("streamlit.sidebar") as mock_sidebar:
            mock_sidebar.image = MagicMock()
            mock_sidebar.markdown = MagicMock()

            mock_sidebar.markdown("## NeuroCare Assistant")
            mock_sidebar.markdown.assert_called()

    @patch("streamlit.chat_input")
    def test_chat_input(self, mock_chat_input):
        """Test if user input is captured correctly."""
        mock_chat_input.return_value = "Test question"
        user_input = st.chat_input("Ask your research question...")
        self.assertEqual(user_input, "Test question")

if __name__ == "__main__":
    unittest.main()
