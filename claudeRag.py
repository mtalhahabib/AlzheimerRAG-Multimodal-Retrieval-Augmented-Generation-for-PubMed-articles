# AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed Articles
import os
import re
import json
# from huggingface_hub import InferenceClient
import shutil
# from huggingface_hub import login

# login("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") 

import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
# from openai import OpenAI
# For text processing
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2

# For vector database
import chromadb
from chromadb.utils import embedding_functions

# For embeddings and LLM
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from PIL import Image
import clip

# For multimodal embeddings
import torch
import clip
import nltk
from chromadb.errors import IDAlreadyExistsError
# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')  # Often needed for text processing
nltk.download('wordnet')   # Sometimes required depending on processing
nltk.download('punkt_tab') # Specific tab tokenization models

# Now add this to handle tab characters in abstracts
nltk.download('popular')   # General safety for common resources

# Set up basic configurations
# class Config:
#     """Configuration for AlzheimerRAG"""
#     # Paths
#     DATA_DIR = "data/"
#     PUBMED_ARTICLES_DIR = os.path.join(DATA_DIR, "pubmed_articles/")
#     PUBMED_IMAGES_DIR = os.path.join(DATA_DIR, "pubmed_images/")
#     DB_DIR = "vectordb/"

#     # Model configs
#     TEXT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Biomedical-specific
#     IMAGE_MODEL = "openai/clip-vit-base-patch32"
#     LLM_MODEL = "meta-llama/Llama-3-8b-hf"  # Replace with your preferred LLM

#     # RAG configs
#     CHUNK_SIZE = 512
#     CHUNK_OVERLAP = 50
#     TOP_K_RESULTS = 5

#     # API configs (if using external APIs)
#     PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")  # Optional
#     OPENAI_API_KEY = "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"  # If using OpenAI models


# Modify the Config class to use a publicly available model
# Replace this in the original code

class Config:
    """Configuration for AlzheimerRAG"""
    # Paths
    DEBUG_MODE = True 
    FALLBACK_MODEL = "Meta-Llama-3-8B-Instruct"
    MIN_SIMILARITY_SCORE = 0.7
    TOP_K_RESULTS = 15
    DATA_DIR = "data/"
    PUBMED_ARTICLES_DIR = os.path.join(DATA_DIR, "pubmed_articles/")
    PUBMED_IMAGES_DIR = os.path.join(DATA_DIR, "pubmed_images/")
    DB_DIR = "vectordb/"

    # Model configs - using publicly available models
    TEXT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # Biomedical-specific
    IMAGE_MODEL = "openai/clip-vit-base-patch32"
    LLM_MODEL = "google/flan-t5-base"  # Public model that doesn't require authentication
    MIN_SIMILARITY_SCORE = 0.7  # Lower threshold
    TOP_K_RESULTS = 10  # Retrieve more candidates
    # RAG configs
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5
    FALLBACK_LLM_API = "meta-llama"  # "openai" or "google"
    OPENAI_API_KEY = "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"  # If using OpenAI models
  # Replace with actual key
    MIN_CONTEXT_LENGTH = 300  # Minimum characters of context needed
    FALLBACK_PROMPT = """You are a medical expert specializing in Alzheimer's disease. 
Provide a comprehensive, evidence-based answer to this query: {query}"""
    FALLBACK_ENABLED = True


# Data structures
@dataclass
class PubMedArticle:
    """Structure for PubMed article data"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    journal: str
    full_text: Optional[str] = None
    figures: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.figures is None:
            self.figures = []

@dataclass
class ArticleChunk:
    """Chunk of text from an article with metadata"""
    chunk_id: str
    text: str
    pmid: str
    title: str
    chunk_type: str  # 'abstract', 'full_text', 'figure_caption'
    section: Optional[str] = None
    page: Optional[int] = None
    figure_id: Optional[str] = None

@dataclass
class QueryResult:
    """Result structure for retrieval"""
    chunks: List[ArticleChunk]
    figures: List[Dict[str, Any]]
    relevance_scores: List[float]

# ----------------------
# 1. DATA COLLECTION AND PROCESSING
# ----------------------

class PubMedFetcher:
    """Fetches articles from PubMed related to Alzheimer's"""

    def __init__(self, api_key=None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = api_key
        os.makedirs(Config.PUBMED_ARTICLES_DIR, exist_ok=True)
        os.makedirs(Config.PUBMED_IMAGES_DIR, exist_ok=True)

    def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed for articles matching query and return PMIDs"""
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(search_url, params=params)
        response_json = response.json()

        pmids = response_json["esearchresult"]["idlist"]
        return pmids

    def fetch_article_details(self, pmid: str) -> Optional[PubMedArticle]:
        """Fetch article metadata for a specific PMID"""
        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(fetch_url, params=params)

        # Simple XML parsing - in a real implementation, use a proper XML parser
        title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", response.text)
        abstract_match = re.search(r"<Abstract>(.*?)</Abstract>", response.text, re.DOTALL)
        journal_match = re.search(r"<Journal>(.*?)</Journal>", response.text, re.DOTALL)
        date_match = re.search(r"<PubDate>(.*?)</PubDate>", response.text, re.DOTALL)

        if not title_match:
            return None

        title = title_match.group(1) if title_match else ""
        abstract = ""
        if abstract_match:
            abstract_text = abstract_match.group(1)
            # Remove XML tags from abstract
            abstract = re.sub(r'<[^>]+>', '', abstract_text)

        journal = journal_match.group(1) if journal_match else ""
        pub_date = date_match.group(1) if date_match else ""

        # Extract authors - simplified
        authors = []
        author_matches = re.finditer(r"<Author>(.*?)</Author>", response.text, re.DOTALL)
        for match in author_matches:
            author_text = match.group(1)
            lastname_match = re.search(r"<LastName>(.*?)</LastName>", author_text)
            firstname_match = re.search(r"<ForeName>(.*?)</ForeName>", author_text)
            if lastname_match:
                author_name = lastname_match.group(1)
                if firstname_match:
                    author_name = f"{firstname_match.group(1)} {author_name}"
                authors.append(author_name)

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            journal=journal
        )

    def download_pdf(self, pmid: str, article: PubMedArticle) -> bool:
        """
        In a real implementation, this would download the full PDF.
        This is a simplified placeholder since PubMed doesn't directly provide PDFs.
        """
        # In practice, you would need to:
        # 1. Find the DOI for the article
        # 2. Navigate to the publisher's website
        # 3. Use institutional access or open access to download
        # 4. This might require Selenium or a similar tool

        # For demonstration, we'll just save the abstract as a text file
        output_path = os.path.join(Config.PUBMED_ARTICLES_DIR, f"{pmid}.txt")
        with open(output_path, "w") as f:
            f.write(f"Title: {article.title}\n\n")
            f.write(f"Authors: {', '.join(article.authors)}\n")
            f.write(f"Journal: {article.journal}\n")
            f.write(f"Date: {article.publication_date}\n\n")
            f.write(f"Abstract:\n{article.abstract}")

        return True

    def extract_figures(self, pmid: str, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract figures from PDF files
        This is a placeholder - extracting images from PDFs requires specialized libraries
        """
        # In a real implementation:
        # 1. Use a library like PyMuPDF (fitz) to extract images
        # 2. Use OCR on the PDF to identify figure captions
        # 3. Match images with their captions

        figures = []
        # Placeholder for demonstration
        figures.append({
            "figure_id": f"{pmid}_fig1",
            "caption": "Figure 1: Sample Alzheimer's brain scan showing amyloid plaques",
            "image_path": os.path.join(Config.PUBMED_IMAGES_DIR, f"{pmid}_fig1.jpg")
        })

        return figures

    def fetch_alzheimer_articles(self, max_articles: int = 50) -> List[PubMedArticle]:
        """Main method to fetch Alzheimer's related articles"""
        query = "Alzheimer's disease[MeSH Terms] AND (\"2020\"[Date - Publication] : \"3000\"[Date - Publication])"
        pmids = self.search_pubmed(query, max_articles)

        articles = []
        for pmid in tqdm(pmids, desc="Fetching articles"):
            article = self.fetch_article_details(pmid)
            if article:
                articles.append(article)
                self.download_pdf(pmid, article)
                # Note: In a real implementation, you would download the actual PDF
                # and extract the full text and figures

        return articles


class TextProcessor:
    """Process and chunk text data from PubMed articles"""

    def __init__(self, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Download NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespaces, special characters, etc."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()
        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")

        return text

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[ArticleChunk]:
        """Split text into chunks with specified size and overlap"""
        if not text:
            return []

        # Split into sentences first
        sentences = sent_tokenize(text)
        chunks = []

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create a new chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{metadata['pmid']}_{len(chunks)}"

                chunks.append(ArticleChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    pmid=metadata['pmid'],
                    title=metadata['title'],
                    chunk_type=metadata.get('chunk_type', 'full_text'),
                    section=metadata.get('section', None),
                    page=metadata.get('page', None),
                    figure_id=metadata.get('figure_id', None)
                ))

                # If we have overlap, keep some sentences for the next chunk
                overlap_tokens = []
                overlap_size = 0

                while current_chunk and overlap_size < self.chunk_overlap:
                    last_sentence = current_chunk.pop()
                    last_size = len(last_sentence)
                    overlap_tokens.insert(0, last_sentence)
                    overlap_size += last_size

                current_chunk = overlap_tokens
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{metadata['pmid']}_{len(chunks)}"

            chunks.append(ArticleChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                pmid=metadata['pmid'],
                title=metadata['title'],
                chunk_type=metadata.get('chunk_type', 'full_text'),
                section=metadata.get('section', None),
                page=metadata.get('page', None),
                figure_id=metadata.get('figure_id', None)
            ))

        return chunks

    def process_article(self, article: PubMedArticle) -> List[ArticleChunk]:
        """Process a PubMed article into chunks"""
        all_chunks = []

        # Process abstract
        if article.abstract:
            abstract_metadata = {
                'pmid': article.pmid,
                'title': article.title,
                'chunk_type': 'abstract'
            }
            abstract_chunks = self.chunk_text(article.abstract, abstract_metadata)
            all_chunks.extend(abstract_chunks)

        # Process full text if available
        if article.full_text:
            fulltext_metadata = {
                'pmid': article.pmid,
                'title': article.title,
                'chunk_type': 'full_text'
            }
            fulltext_chunks = self.chunk_text(article.full_text, fulltext_metadata)
            all_chunks.extend(fulltext_chunks)

        # Process figure captions
        for figure in article.figures or []:
            if 'caption' in figure:
                figure_metadata = {
                    'pmid': article.pmid,
                    'title': article.title,
                    'chunk_type': 'figure_caption',
                    'figure_id': figure['figure_id']
                }
                caption_chunks = self.chunk_text(figure['caption'], figure_metadata)
                all_chunks.extend(caption_chunks)

        return all_chunks


# ----------------------
# 2. EMBEDDING GENERATION
# ----------------------

class EmbeddingGenerator:
    """Generate embeddings for text and images"""

    def __init__(self):
        # Load text embedding model (biomedical-specific)
        self.text_embedding_model = SentenceTransformer(Config.TEXT_EMBEDDING_MODEL)

        # Load CLIP for image embeddings
        self.image_model, self.image_preprocess = clip.load(
            "ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.text_embedding_model.encode(text)
        return embedding.tolist()

    def generate_image_embedding(self, image_path: str) -> List[float]:
        """Generate embedding for image"""
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = self.image_preprocess(image).unsqueeze(0)

            # Generate embedding
            with torch.no_grad():
                image_features = self.image_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 512  # CLIP embeddings are 512-dimensional

    def process_article_chunks(self, chunks: List[ArticleChunk]) -> Dict[str, List[float]]:
        """Generate embeddings for a list of article chunks"""
        embeddings = {}
        for chunk in tqdm(chunks, desc="Generating text embeddings"):
            embeddings[chunk.chunk_id] = self.generate_text_embedding(chunk.text)
        return embeddings

    def process_article_figures(self, article: PubMedArticle) -> Dict[str, List[float]]:
        """Generate embeddings for figures in an article"""
        embeddings = {}
        for figure in article.figures or []:
            if 'image_path' in figure and os.path.exists(figure['image_path']):
                figure_id = figure['figure_id']
                embeddings[figure_id] = self.generate_image_embedding(figure['image_path'])
        return embeddings


# ----------------------
# 3. VECTOR DATABASE
# ----------------------

class VectorDatabase:
    """Vector database for storing and retrieving embeddings"""

    def __init__(self, db_dir=Config.DB_DIR):
        self.db_dir = db_dir
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)
        
        os.makedirs(db_dir, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_dir,
            settings=chromadb.Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        # Create text collection
        self.text_collection = self.client.get_or_create_collection(
            name="alzheimer_text",
            metadata={"description": "Text chunks from Alzheimer's PubMed articles"}
        )

        # Create image collection
        self.image_collection = self.client.get_or_create_collection(
            name="alzheimer_images",
            metadata={"description": "Images from Alzheimer's PubMed articles"}
        )

        # Store chunk data separately (ChromaDB doesn't store the full data efficiently)
        self.chunks_store = {}
        self.figures_store = {}

    # def add_text_chunks(self, chunks: List[ArticleChunk], embeddings: Dict[str, List[float]]):
    #     """Add text chunks to the database"""
    #     ids = []
    #     embedding_vectors = []
    #     metadatas = []

    #     for chunk in chunks:
    #         chunk_id = chunk.chunk_id
    #         if chunk_id not in embeddings:
    #             continue

    #         ids.append(chunk_id)
    #         embedding_vectors.append(embeddings[chunk_id])
    #         metadatas.append({
    #             "pmid": chunk.pmid,
    #             "title": chunk.title,
    #             "chunk_type": chunk.chunk_type,
    #             "section": chunk.section if chunk.section else "",
    #             "figure_id": chunk.figure_id if chunk.figure_id else ""
    #         })

    #         # Store the full chunk data
    #         self.chunks_store[chunk_id] = chunk

    #     if ids:
    #         # Add to ChromaDB in batches
    #         batch_size = 100
    #         for i in range(0, len(ids), batch_size):
    #             end = min(i + batch_size, len(ids))
    #             print('>')
    #             print('iiiii${i}')
    #             self.text_collection.add(
    #                 ids=ids[i:end],
    #                 embeddings=embedding_vectors[i:end],
    #                 metadatas=metadatas[i:end]
    #             )

    def add_text_chunks(self, chunks: List[ArticleChunk], embeddings: Dict[str, List[float]]):
        """Add text chunks to the database"""
        
        
        ids = []
        embedding_vectors = []
        metadatas = []

        # Get existing IDs to avoid duplicates
        existing_ids = set(self.text_collection.get(include=[])['ids'])
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            if chunk_id not in embeddings or chunk_id in existing_ids:
                continue

            ids.append(chunk_id)
            embedding_vectors.append(embeddings[chunk_id])
            metadatas.append({
                "pmid": chunk.pmid,
                "title": chunk.title,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section if chunk.section else "",
                "figure_id": chunk.figure_id if chunk.figure_id else ""
            })
            self.chunks_store[chunk_id] = chunk

        if ids:
            batch_size = 100
            total_batches = (len(ids) + batch_size - 1) // batch_size
            
            for batch_idx, i in enumerate(range(0, len(ids), batch_size)):
                end = min(i + batch_size, len(ids))
                try:
                    print(f"Processing batch {batch_idx+1}/{total_batches}")
                    self.text_collection.add(
                        ids=ids[i:end],
                        embeddings=embedding_vectors[i:end],
                        metadatas=metadatas[i:end]
                    )
                except IDAlreadyExistsError as e:
                    print(f"Skipped duplicate batch {batch_idx+1}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    raise

    def add_figures(self, article: PubMedArticle, figure_embeddings: Dict[str, List[float]]):
        """Add figures to the database"""
        ids = []
        embedding_vectors = []
        metadatas = []

        for figure in article.figures or []:
            figure_id = figure['figure_id']
            if figure_id not in figure_embeddings:
                continue

            ids.append(figure_id)
            embedding_vectors.append(figure_embeddings[figure_id])
            metadatas.append({
                "pmid": article.pmid,
                "title": article.title,
                "caption": figure.get('caption', '')
            })

            # Store the full figure data
            self.figures_store[figure_id] = figure

        if ids:
            # Add to ChromaDB in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                self.image_collection.add(
                    ids=ids[i:end],
                    embeddings=embedding_vectors[i:end],
                    metadatas=metadatas[i:end]
                )

    def retrieve_similar_texts(self, query_embedding: List[float], top_k: int = Config.TOP_K_RESULTS) -> List[Tuple[str, float]]:
        """Retrieve text chunks similar to the query embedding"""
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas"]
        )

        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]

        # Convert distances to similarity scores (1 - distance)
        similarity_scores = [1 - dist for dist in distances]

        return list(zip(ids, similarity_scores))

    def retrieve_similar_images(self, query_embedding: List[float], top_k: int = Config.TOP_K_RESULTS) -> List[Tuple[str, float]]:
        """Retrieve images similar to the query embedding"""
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas"]
        )

        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]

        # Convert distances to similarity scores (1 - distance)
        similarity_scores = [1 - dist for dist in distances]

        return list(zip(ids, similarity_scores))

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ArticleChunk]:
        """Get a text chunk by its ID"""
        return self.chunks_store.get(chunk_id)

    def get_figure_by_id(self, figure_id: str) -> Optional[Dict[str, Any]]:
        """Get a figure by its ID"""
        return self.figures_store.get(figure_id)


# ----------------------
# 4. RAG SYSTEM
# ----------------------

class AlzheimerRAG:
    """Multimodal RAG system for Alzheimer's research articles"""

    # def __init__(self, llm_model=Config.LLM_MODEL):
    #     self.embedding_generator = EmbeddingGenerator()
    #     self.vector_db = VectorDatabase()

    #     # Initialize LLM for generation
    #     self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
    #     self.pipeline = pipeline(
    #         "text-generation",
    #         model=llm_model,
    #         tokenizer=self.tokenizer,
    #         max_new_tokens=512,
    #         temperature=0.7,
    #         top_p=0.9
    #     )

    def __init__(self, llm_model=Config.LLM_MODEL):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()

        # Initialize LLM for generation - using T5 instead of Llama
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        # In AlzheimerRAG.__init__() modify the pipeline creation:
        self.pipeline = pipeline(
            "text2text-generation",
            model=llm_model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True,  # Add this to enable temperature
            top_p=0.9        # Add nucleus sampling
        )
        # if Config.FALLBACK_LLM_API == "meta-llama":
        #     # self.meta_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        #     self.meta_client = InferenceClient(model='meta-llama/Meta-Llama-3-8B-Instruct', api_key='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')

    def generate(self, query: str, retrieval_results: QueryResult) -> str:
        # Construct context with proper T5 formatting
        context = "\n".join([
        f"Study Title: {chunk.title}\n"
        f"PMID: {chunk.pmid}\n"
        f"Content: {chunk.text}\n" 
        for chunk in retrieval_results.chunks
    ])
    
        prompt = f"""As an Alzheimer's research specialist, synthesize this information:
    {context}
    
    Provide a structured response with:
    1. Key pathological findings
    2. Diagnostic approaches 
    3. Therapeutic strategies
    4. Research challenges
    
    Question: {query}
    Answer in clear, clinical language:"""
        response = self.pipeline(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]['generated_text']
        
        return response
    def index_articles(self, articles: List[PubMedArticle]):
        """Index articles for retrieval"""
        import gc
        text_processor = TextProcessor()
        print(len(articles))
        for article in tqdm(articles, desc="Indexing articles"):
            # Process text into chunks
            chunks = text_processor.process_article(article)

            # Generate text embeddings
            chunk_embeddings = self.embedding_generator.process_article_chunks(chunks)

            # Generate image embeddings
            figure_embeddings = self.embedding_generator.process_article_figures(article)

            # Add to vector database
            print(4)
            self.vector_db.add_text_chunks(chunks, chunk_embeddings)
            print(5.5)
            self.vector_db.add_figures(article, figure_embeddings)
            print(6)
            del chunks
            del chunk_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            print(7)  # Add garbage collection
            if self.vector_db.text_collection.count() == 0:
                print("Performing initial indexing...")

                self._force_reindex(articles)

            print(8)
            print(9)
        print(9.5)
        total_indexed = self.vector_db.text_collection.count()
        print(10)
        print(f"Successfully indexed {total_indexed} text chunks")
        
        fig_count = self.vector_db.image_collection.count()
        print(f"Successfully indexed {fig_count} images")

    def _force_reindex(self, articles):
        self.vector_db.client.reset()
        # Recreate collections
        self.vector_db = VectorDatabase() 
        # Reprocess articles
        self.index_articles(articles)

    def retrieve(self, query: str, top_k: int = Config.TOP_K_RESULTS, images: bool = True) -> QueryResult:
        """Retrieve relevant information for the query with enhanced quality control"""
        # Add similarity threshold to filter irrelevant results
        MIN_SIMILARITY_SCORE = 0.4  # Adjust based on your embedding space
        MAX_IMAGE_RESULTS = 3  # Separate control for image results

        # Generate query embedding with error handling
        try:
            expanded_query = f"{query} Alzheimer's disease dementia cognitive impairment"
            query_embedding = self.embedding_generator.generate_text_embedding(expanded_query)
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            return QueryResult([], [], [])

        # Retrieve and filter text chunks
        raw_text_results = self.vector_db.retrieve_similar_texts(query_embedding, top_k * 2)  # Over-fetch to allow filtering
        filtered_text_results = [
            (chunk_id, score) 
            for chunk_id, score in raw_text_results
            if score >= MIN_SIMILARITY_SCORE
        ][:top_k]  # Final top_k after filtering
        # Add after similarity filtering
        if not filtered_text_results:
            return QueryResult([], [], [])
        # Remove duplicate chunks (same PMID)
        seen_pmids = set()
        retrieved_chunks = []
        relevance_scores = []
        for chunk_id, score in filtered_text_results:
            chunk = self.vector_db.get_chunk_by_id(chunk_id)
            if chunk and chunk.pmid not in seen_pmids:
                retrieved_chunks.append(chunk)
                relevance_scores.append(score)
                seen_pmids.add(chunk.pmid)

        # Image retrieval with separate quality control
        retrieved_figures = []
        if images:
            raw_image_results = self.vector_db.retrieve_similar_images(query_embedding, top_k=MAX_IMAGE_RESULTS * 2)
            filtered_image_results = [
                (figure_id, score)
                for figure_id, score in raw_image_results
                if score >= MIN_SIMILARITY_SCORE
            ][:MAX_IMAGE_RESULTS]

            seen_figure_ids = set()
            for figure_id, score in filtered_image_results:
                figure = self.vector_db.get_figure_by_id(figure_id)
                if figure and figure['figure_id'] not in seen_figure_ids:
                    retrieved_figures.append(figure)
                    relevance_scores.append(score)
                    seen_figure_ids.add(figure['figure_id'])

        # Normalize scores for consistency
        if relevance_scores:
            max_score = max(relevance_scores)
            relevance_scores = [score/max_score for score in relevance_scores]

        # Sort results by descending relevance
        sorted_indices = sorted(
            range(len(relevance_scores)), 
            key=lambda i: -relevance_scores[i]
        )
        if Config.DEBUG_MODE:  # Add DEBUG_MODE to Config class
            print(f"\n🔎 Retrieval Report for: '{query}'")
            print(f"📚 Text Candidates: {len(raw_text_results)}")
            print(f"🖼️ Image Candidates: {len(raw_image_results)}")
            print(f"✅ Filtered Texts: {len(retrieved_chunks)}")
            print(f"✅ Filtered Images: {len(retrieved_figures)}")
            
            if retrieved_chunks:
                print("\nTop Text Matches:")
                for idx, chunk in enumerate(retrieved_chunks[:3]):
                    print(f"{idx+1}. {chunk.title} (PMID: {chunk.pmid})")
                    print(f"   {chunk.text[:100]}...")
                    
            if retrieved_figures:
                print("\nTop Images:")
                for idx, fig in enumerate(retrieved_figures[:3]):
                    print(f"{idx+1}. {fig.get('caption','No caption')[:80]}...")

        return QueryResult(
            chunks=[retrieved_chunks[i] for i in sorted_indices if i < len(retrieved_chunks)],
            figures=[retrieved_figures[i] for i in sorted_indices if i >= len(retrieved_chunks)],
            relevance_scores=[relevance_scores[i] for i in sorted_indices]
        )
#     def generate(self, query: str, retrieval_results: QueryResult) -> str:
#         """Generate a response based on the query and retrieved information"""
#         # Construct context from retrieved chunks
#         context = ""
#         for chunk in retrieval_results.chunks:
#             context += f"### Article: {chunk.title} (PMID: {chunk.pmid})\n"
#             context += f"{chunk.text}\n\n"

#         # Add figure information if available
#         for figure in retrieval_results.figures:
#             context += f"### Figure: {figure.get('caption', 'Untitled figure')}\n"
#             context += f"From article PMID: {figure.get('pmid', 'Unknown')}\n\n"

#         # Construct the prompt
#         prompt = f"""You are an expert on Alzheimer's disease research. Using the following information from scientific articles, please answer the question.

# Information from PubMed articles:
# {context}

# Question: {query}

# Please provide a comprehensive and scientifically accurate answer based on the provided information:"""

#         # Generate response
#         response = self.pipeline(prompt)[0]['generated_text']

#         # Extract the generated part (after the prompt)
#         generated_text = response[len(prompt):].strip()

#         return generated_text
# Add to Config class
    def answer(self, query: str, include_images: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
            """Answer a query using RAG with improved validation"""
            try:
                # Retrieve relevant information
                # retrieval_results = self.retrieve(query, images=include_images)
                
                # # Calculate context adequacy with debug logging
                # total_context = sum(len(chunk.text) for chunk in retrieval_results.chunks)
                # print(f"Retrieved context length: {total_context} characters")
                
                # # Generate base response if any context exists
                # if retrieval_results.chunks:
                #     print('retrieveal results')
                #     response = self.generate(query, retrieval_results)
                #     if self._validate_response(response, retrieval_results):
                #         return response, self._process_figures(retrieval_results)
                
                # # Fallback only if enabled and no context
                # if Config.FALLBACK_ENABLED:
                #     print("Falling back to external LLM")
                prompt = f"""**Take Context from Recent Research on Alzheimers Disease**


                **Patient Question:**
                {query}

                Please provide a compassionate, evidence-based response as an Alzheimer's specialist and an Assistant:"""

                payload = {
                            "messages":[
                                {
                                    "role":"user",
                                    "content":prompt
                                }
                            ],
                            "model": "meta-llama/llama-3.1-8b-instruct"
                        
                        }
                print('wait')
                response = requests.post(
                            "https://router.huggingface.co/novita/v3/openai/chat/completions",
                            headers={"Authorization": "Bearer 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"},
                            json=payload
                        )
                # print(response.status_code)
                # # if response.status_code == 200:
                #             # Clean up any residual instructions from response
                # print(response)
                # print(response.json())
                                # print(response.json["choices"][0]["message"]['content'])
                full_response = response.json()
                res=full_response['choices'][0]['message']['content']
                
                                # Remove any remaining system messages
                # clean_response = full_response.split("Please provide")[0].strip()
                return res,[]
                # return self._openai_fallback(query), []
                
                # return "I couldn't find relevant research. Please try rephrasing your question.", []
            
            except Exception as e:
                print(f"Critical error: {str(e)}")
                return self._safe_fallback_response(query), []

    def _validate_response(self, response: str, results: QueryResult) -> bool:
            """Improved validation with context awareness"""
            # Basic length check
            if len(response.strip()) < 100:
                print("Validation failed: Response too short")
                return False
                
            # Check for presence of key terms from query
            query_terms = set(word.lower() for word in re.findall(r'\w+', query))
            response_terms = set(word.lower() for word in re.findall(r'\w+', response))
            
            if not query_terms.intersection(response_terms):
                print(f"Validation failed: Missing query terms {query_terms}")
                return False
                
            # Check if response contains information from context
            context_phrases = set()
            for chunk in results.chunks:
                context_phrases.update(chunk.text.split()[:10])  # First 10 words
                
            return any(phrase in response for phrase in context_phrases)

    def _openai_fallback(self, query: str) -> str:
        """Improved fallback with context-aware prompting"""
        try:
            # Retrieve context from internal knowledge base
            # general_results = self.retrieve("Alzheimer's disease general information", 3)
            # context = "\n".join([chunk.text for chunk in general_results.chunks[:3]])

        # Revised prompt structure
            prompt = f"""**Take Context from Recent Research on Alzheimers Disease**


**Patient Question:**
{query}
Answer the question in a clear, natural tone. Do not include titles, headings, or formal salutations like "Dear Patient". Just respond like an Alzheimers assistant bot.
"""

            payload = {
            "messages":[
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            "model": "meta-llama/llama-3.1-8b-instruct"
        
        }
            print('wait')
            response = requests.post(
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
            headers={"Authorization": "Bearer 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"},
            json=payload
        )
            print(response.status_code)
            if response.status_code == 200:
            # Clean up any residual instructions from response
                print(response)
                print(response.json)
                # print(response.json["choices"][0]["message"]['content'])
                full_response = response.json()[0]["generated_text"]
                print(full_response)
                # Remove any remaining system messages
                clean_response = full_response.split("Please provide")[0].strip()
                return clean_response
         
            elif response.status_code == 404:
                print("❌ Model repository not found! Check if the model exists or is private.")
            elif response.status_code == 503:
                print("⚠️ Model is currently overloaded. Retrying with OpenAI fallback...")
            else:
                print(f"❌ API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"⚠️ Fallback failed: {str(e)}")

        # Default response in case of failure
        return "Current research suggests consulting a healthcare professional for personalized advice."

    def _process_figures(self, results: QueryResult) -> List[Dict[str, Any]]:
            """Process figures for response"""
            return [{
                'id': fig.get('figure_id', ''),
                'caption': fig.get('caption', ''),
                'path': fig.get('image_path', '')
            } for fig in results.figures]

    def _safe_fallback_response(self, query):
        return f"""I encountered an error processing your query about {query}. 
        Current research suggests consulting peer-reviewed journals like 
        'Alzheimer's & Dementia' for the latest findings."""
        
# ----------------------
# 5. EXAMPLE USAGE
# ----------------------

def run_alzheimer_rag_example():
    """Example of running the AlzheimerRAG system"""
    # 1. Initialize the fetcher
    fetcher = PubMedFetcher()

    # 2. Fetch Alzheimer's articles (limited to 10 for this example)
    print("Fetching Alzheimer's articles from PubMed...")
    articles = fetcher.fetch_alzheimer_articles(max_articles=10)
    print(f"Retrieved {len(articles)} articles")

    # 3. Initialize the RAG system
    rag_system = AlzheimerRAG()

    # 4. Index the articles
    print("Indexing articles...")
    rag_system.index_articles(articles)

    # 5. Example queries
    queries = [
        "What are the latest findings on amyloid beta in Alzheimer's disease?",
        "How does tau protein contribute to neurodegeneration in Alzheimer's?",
        "What imaging biomarkers are used for early Alzheimer's detection?"
    ]

    # 6. Answer queries
    for query in queries:
        print(f"\nQuery: {query}")
        response, figures = rag_system.answer(query)
        print("\nResponse:")
        print(response)

        if figures:
            print("\nRelevant Figures:")
            for fig in figures:
                print(f"- {fig['caption']} (from PMID: {fig['pmid']})")

def run_diagnostic_tests():
    """Example of running the AlzheimerRAG system"""
    # 1. Initialize the fetcher
    fetcher = PubMedFetcher()

    # 2. Fetch Alzheimer's articles (limited to 10 for this example)
    
    print("Fetching Alzheimer's articles from PubMed...")
    articles = fetcher.fetch_alzheimer_articles(max_articles=10)
    print(f"Retrieved {len(articles)} articles")

    # 3. Initialize the RAG system
    rag_system = AlzheimerRAG()

    # 4. Index the articles
    print("Indexing articles...")
    rag_system.index_articles(articles)
    test_cases = [
            ("Basic Definition", "What is amyloid beta?"),
            ("Protein Query", "How does tau protein contribute to Alzheimer's?"),
            ("Diagnostic Methods", "What imaging biomarkers are used for diagnosis?"),
            ("Genetic Factors", "What role does APOE ε4 play in Alzheimer's risk?"),
            ("Treatment Query", "What are current pharmacological interventions?")
        ]
        
    for test_name, query in test_cases:
            print(f"\n🧪 Test Case: {test_name}")
            print(f"📝 Query: {query}")
            start_time = time.time()
            
            response, figures = rag_system.answer(query)
            
            print(f"⏱️ Response Time: {time.time()-start_time:.2f}s")
            print(f"📄 Response Length: {len(response)} characters")
            print(f"🖼️ Figures Found: {len(figures)}")
            print("\n" + response[:500] + "...\n")


if __name__ == "__main__":
    # run_alzheimer_rag_example()
    run_diagnostic_tests()


# ----------------------
# 6. WEB INTERFACE (OPTIONAL)
# ----------------------

def create_web_interface():
    """
    Optional: Create a web interface for the AlzheimerRAG system
    This would use Flask or Streamlit in a complete implementation
    """
    # Example of what a Flask app would look like
    """
    from flask import Flask, request, jsonify, render_template

    app = Flask(__name__)
    rag_system = AlzheimerRAG()

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/api/answer', methods=['POST'])
    def answer():
        data = request.json
        query = data.get('query', '')
        include_images = data.get('include_images', True)

        response, figures = rag_system.answer(query, include_images)

        return jsonify({
            'response': response,
            'figures': figures
        })

    if __name__ == '__main__':
        # Load pre-indexed data
        app.run(debug=True)
    """
    pass