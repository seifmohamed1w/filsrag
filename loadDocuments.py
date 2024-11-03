import os
import logging
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import re
from uuid import uuid4

logging.basicConfig(level=logging.INFO)

# Define the directory to persist the embeddings database and where documents are located
directory_path = r'C:\\Users\\seif\\Desktop\\filschatbot\\UpbFils'
persist_directory = 'db'
embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# Load the Hugging Face model for semantic analysis
model = SentenceTransformer(embedding_model_name)


# Text cleaning function to remove excessive spaces and other formatting issues
def clean_text(text):
    # Normalize space: replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return text.strip()


# Rule-based splitting using regex
def rule_based_split(text):
    return re.split(r'\n(?:\d+\.\s|\d+\)\s|•\s|\-\s)?(?=[A-Z])', text)


# Semantic splitting using embeddings and clustering
def semantic_splitting(paragraphs):
    embeddings = model.encode(paragraphs, show_progress_bar=True)
    num_clusters = max(1, int(len(paragraphs) ** 0.5))
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(embeddings)

    inner_chunks = []
    current_chunk = paragraphs[0]
    for idx in range(1, len(paragraphs)):
        if labels[idx] != labels[idx - 1]:
            inner_chunks.append(current_chunk)
            current_chunk = paragraphs[idx]
        else:
            current_chunk += '\n\n' + paragraphs[idx]
    inner_chunks.append(current_chunk)
    return inner_chunks


# Hybrid splitting
def hybrid_splitting(doc):
    # Use the 'page_content' attribute to access the document's text
    cleaned_text = clean_text(doc.page_content)
    initial_chunks = rule_based_split(cleaned_text)
    refined_chunks = []
    for chunk in initial_chunks:
        refined_chunks.extend(semantic_splitting(chunk.split('\n\n')))
    return refined_chunks


# Load and process the text files using PyPDFDirectoryLoader
try:
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    logging.info(f"Successfully loaded {len(documents)} documents.")
except Exception as e:
    logging.error(f"Failed to load documents from directory: {e}")
    raise

# Process each document into LangChain Document objects
doc_objects = []
try:
    for doc in tqdm(documents, desc="Splitting documents into chunks"):
        inner_chunks = hybrid_splitting(doc)
        for chunk in inner_chunks:
            doc_objects.append(Document(
                page_content=chunk,
                metadata={'source': doc.metadata.get('source', 'unknown')}
            ))
    logging.info(f"Successfully split documents into {len(doc_objects)} chunks.")
except Exception as e:
    logging.error(f"Error during text splitting: {e}")
    raise

# Generate UUIDs for each document
uuids = [str(uuid4()) for _ in range(len(doc_objects))]

# Initialize Chroma with hybrid retrieval capabilities and add documents with UUIDs
try:
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma(collection_name="example_collection", embedding_function=embedding,
                      persist_directory=persist_directory)

    # Add documents with generated UUIDs
    vectordb.add_documents(documents=doc_objects, ids=uuids)

    # Persist the vector store to disk
    vectordb.persist()
    logging.info("Vector database created and persisted successfully.")

except Exception as e:
    logging.error(f"Error during vector database creation and persistence: {e}")
    raise
