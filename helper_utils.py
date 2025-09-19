import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
from tqdm import tqdm


def _read_pdf(filename):
    reader = PdfReader(filename)
    
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def load_chroma(filename, collection_name, embedding_function):
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)

    chroma_cliet = chromadb.Client()
    chroma_collection = chroma_cliet.create_collection(name=collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]

    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)
    
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def load_chroma_persist(filename,collection_name,embedding_function,dir_path = "./chroma_db" , force_create = False):
    chroma_client = chromadb.PersistentClient(path=dir_path)
    try:
        chroma_collection = chroma_client.get_collection(name=collection_name,embedding_function=embedding_function)
        print(f"Loading the existing chroma collection {collection_name}")
        return chroma_collection
    
    except Exception:
        print(f"creating new collection {collection_name}")
        print("📖 Reading PDF...")
        texts = _read_pdf(filename)
        print("✂️ Chunking texts...")
        chunks = _chunk_texts(texts)
        ids = [str(i) for i in range(len(chunks))]
        chroma_collection = chroma_client.create_collection(name=collection_name,embedding_function=embedding_function)
        batch_size = 50        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            chroma_collection.add(
                ids=batch_ids,
                documents=batch_chunks
            )
            
        print("✅ Collection created successfully!")
        return chroma_collection


if __name__ == "__main__":
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    filename = "microsoft_annual_report_2022.pdf"
    collection_name = "microsoft_annual_report_2022"
    emb_func = SentenceTransformerEmbeddingFunction()
    load_chroma_persist(filename,collection_name,emb_func)
    
