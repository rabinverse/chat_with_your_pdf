from langchain_community.vectorstores import FAISS, Chroma
from load_docs_embeddingmodel_setup import get_embedding_model, chunked_documents
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
import shutil

embedding_model = get_embedding_model()


def backup_path(path):
    if os.path.exists(path):
        bak = f"{path}_backup"
        if os.path.exists(bak):
            shutil.rmtree(bak)
        shutil.copytree(path, bak)
        return bak
    return None


def faiss_local(db_faiss_path="vectorstore/db_faiss", documents=None):
    """
    Behavior:
    - If DB does not exist -> create from `documents` (or chunked_documents) and save_local
    - If DB exists -> ask user: replace / append / load / cancel
      - replace -> recreate from documents and save_local
      - append -> load existing, add_documents(new_docs), save_local
      - load -> simply load and return
    """
    documents = documents if documents is not None else chunked_documents
    parent = os.path.dirname(db_faiss_path) or "."
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(db_faiss_path):
        choice = (
            input(
                f"FAISS DB exists at '{db_faiss_path}'. Choose: [replace | append | load | cancel]: "
            )
            .strip()
            .lower()
        )

        if choice == "replace":
            backup_path(db_faiss_path)
            db = FAISS.from_documents(documents, embedding_model)
            db.save_local(db_faiss_path)
            return db

        if choice == "append":
            db = FAISS.load_local(
                db_faiss_path, embedding_model, allow_dangerous_deserialization=True
            )
            db.add_documents(documents)
            db.save_local(db_faiss_path)
            return db

        if choice == "load":
            db = FAISS.load_local(
                db_faiss_path, embedding_model, allow_dangerous_deserialization=True
            )
            return db

        # cancel or anything else
        print("Operation cancelled.")
        return None

    # does not exist -> create
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(db_faiss_path)
    return db


def chroma_local(db_chroma_path="vectorstore/db_chroma", documents=None):
    """
    Behavior:
    - If DB does not exist -> create and persist
    - If DB exists -> ask user: replace / append / load / cancel
      - replace -> backup + delete directory, create from documents, persist
      - append -> load existing Chroma, add_documents(new_docs), persist
      - load -> return loaded Chroma instance
    """
    documents = documents if documents is not None else chunked_documents
    # ensure parent directory exists
    parent = os.path.dirname(db_chroma_path) or "."
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(db_chroma_path):
        choice = (
            input(
                f"Chroma DB exists at '{db_chroma_path}'. Choose: [replace | append | load | cancel]: "
            )
            .strip()
            .lower()
        )

        if choice == "replace":
            bak = backup_path(db_chroma_path)
            shutil.rmtree(db_chroma_path)
            db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=db_chroma_path,
            )
            db.persist()
            return db

        if choice == "append":
            # load existing chroma and add new docs
            db = Chroma(
                persist_directory=db_chroma_path, embedding_function=embedding_model
            )
            db.add_documents(documents)
            db.persist()
            return db

        if choice == "load":
            db = Chroma(
                persist_directory=db_chroma_path, embedding_function=embedding_model
            )
            return db

        print("Operation cancelled.")
        return None

    # does not exist -> create
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=db_chroma_path,
    )
    db.persist()
    return db


def pinecone_local(
    index_name="pdf-vector-index",
    documents=None,
    dimension=768,  # must match embedding model
    metric="cosine",
):
    """
    Behavior:
    - If index does NOT exist -> create + upsert documents
    - If index exists:
        - ask user: replace / append / load / cancel
        - replace -> delete index, recreate, upsert
        - append -> upsert new documents
        - load -> load existing index
    """

    documents = documents if documents is not None else chunked_documents

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        choice = (
            input(
                f"Pinecone index '{index_name}' exists. Choose: "
                "[replace | append | load | cancel]: "
            )
            .strip()
            .lower()
        )

        if choice == "replace":
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embedding_model,
                index_name=index_name,
            )
            return vectorstore

        if choice == "append":
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding_model,
            )
            vectorstore.add_documents(documents)
            return vectorstore

        if choice == "load":
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding_model,
            )
            return vectorstore

        print("Operation cancelled.")
        return None

    # index does not exist → create
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        index_name=index_name,
    )
    return vectorstore
