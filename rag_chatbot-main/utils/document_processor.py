# /utils/document_processor.py

import os
import logging
from glob import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

def load_and_split_pdfs():
    """
    Loads PDFs from the directory specified in config and splits them into chunks.

    Returns:
        list: A list of document chunks (LangChain Document objects).
    """
    doc_path = config.DOCUMENT_DIRECTORY
    logging.info(f"Loading documents from '{doc_path}'...")

    if not os.path.isdir(doc_path):
        logging.warning(f"Document directory '{doc_path}' not found. No documents will be loaded.")
        return []

    pdf_files = glob(os.path.join(doc_path, "*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDF files found in '{doc_path}'. Skipping knowledge base setup.")
        return []

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    for path in pdf_files:
        try:
            logging.info(f"Processing {os.path.basename(path)}...")
            loader = PyPDFLoader(path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            logging.info(f"Split {os.path.basename(path)} into {len(chunks)} chunks.")
        except Exception as e:
            logging.error(f"Error processing file {path}: {e}")

    if all_chunks:
        logging.info(f"Total foundational chunks created: {len(all_chunks)}")

    return all_chunks