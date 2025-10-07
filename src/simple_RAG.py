from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import glob, json
from tqdm import tqdm

class SimpleRAG:
    """
    SimpleRAG handles loading, chunking, and embedding reference documents into a vector store.
    """

    def __init__(self, ref_data_directory, vector_store):
        """
        Initialize with reference data directory and a vector store instance.

        Args:
            ref_data_directory (str): Path to the root directory containing reference data.
            vector_store: Vector store object with an add_documents method.
        """
        self.ref_data_directory = ref_data_directory
        self.vector_store = vector_store
        self.data_directories = [
            self.ref_data_directory + "Financials/",
            self.ref_data_directory + "HR/"
        ]
        self.files_to_embed = []

    def collect_files(self):
        """
        Collect all PDF, JSON, and TXT files from the specified data directories.
        """
        for directory in self.data_directories:
            files = (
                glob.glob(directory + "*.pdf") +
                glob.glob(directory + "*.json") +
                glob.glob(directory + "*.txt")
            )
            self.files_to_embed.extend(files)

    def print_files(self):
        """
        Print the list of files that will be embedded.
        """
        print("List of files_to_embed:")
        for file in self.files_to_embed:
            print(file)

    def embed_files(self):
        """
        Load, split, and embed each collected file into the vector store.
        """
        for file in tqdm(self.files_to_embed):
            loader = PyPDFLoader(file)
            docs = loader.load()
            # Split documents into chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
            all_splits = text_splitter.split_documents(docs)
            # Add document chunks to the vector store
            _ = self.vector_store.add_documents(documents=all_splits)
            print(f"Vector Store: Indexed {len(all_splits)} chunks from {file} | {_}")

    def run(self):
        """
        Execute the full pipeline: collect files, print them, and embed them.
        """
        self.collect_files()
        self.print_files()
        self.embed_files()
