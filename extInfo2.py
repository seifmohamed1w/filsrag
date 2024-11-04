import os
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# Set the OpenAI API key securely
os.environ["OPENAI_API_KEY"] = "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"

# Define the directory to persist the database
persist_directory = 'db'
embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load the persisted database
vectordb = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory=persist_directory
)

# Create a retriever
retriever = vectordb.as_retriever()

# Create the QA chain using OpenAI LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(max_tokens=500,model="gpt-3.5-turbo-instruct"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


def get_top_chunks_with_scores(user_input, top_n=5):
    """Retrieve the top N chunks with their similarity scores using similarity_search_with_score."""
    # Retrieve documents with similarity scores
    retrieved_chunks_with_scores = retriever.similarity_search_with_score(user_input, k=50)

    # Sort and select the top N chunks
    top_chunks = sorted(retrieved_chunks_with_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # Extract the document objects from the top N results
    top_chunks = [chunk[0] for chunk in top_chunks]

    return top_chunks


def process_user_input(user_input):
    """Process user input by retrieving and selecting top-ranked chunks, then generating a response."""
    # Retrieve top chunks with scores
    top_chunks = get_top_chunks_with_scores(user_input)

    # Combine the content of the top chunks for the final input to the QA chain
    combined_context = " ".join([chunk.page_content for chunk in top_chunks])

    # Generate the response using the QA chain with the filtered context
    llm_response = qa_chain({"query": user_input, "context": combined_context})
    return process_llm_response(llm_response)


def process_llm_response(llm_response):
    """Process the LLM response and add details if it seems too brief."""
    response = llm_response['result']
    if len(response.split()) < 50:  # Example condition for short responses
        response += "\n\nCan you provide more details?"
    return response


def store_feedback(user_input, response, feedback):
    """Stores user feedback in the SQLite database."""
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_input, response, user_feedback) VALUES (?, ?, ?)",
              (user_input, response, feedback))
    conn.commit()
    conn.close()
