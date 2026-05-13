# Extracts qdrant Vector Store Database containing our text documentation
# into with 2 column dataframe where 
# (prompt: title of the page and completion: the contents of the page)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import qdrant_client.http.exceptions as qdrant_exceptions
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import os
import pandas as pd

qdrant_path = "./langchain_qdrant"
qdrant_collection_name = "chpc-rag"
qdrant_vector_size = 768
qdrant_distance = Distance.COSINE
embeddings_model_name = "all-mpnet-base-v2"

def setup_qdrant():   
   try:
        qc = QdrantClient(path=qdrant_path)
        cols = qc.get_collections()
        have = [c.name for c in cols.collections]
        create = qdrant_collection_name not in have
        qc.close()
   except Exception as e:
        # logger.error("Qdrant initial probe failed", exc_info=True)
        raise RuntimeError(f"Qdrant probe failed: {e}")

   embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

   if create:
        dim = len(embeddings.embed_query("dimension probe"))
        qc2 = QdrantClient(path=qdrant_path)
        qc2.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=VectorParams(size=dim, distance=qdrant_distance),
        )
        qc2.close()

   return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=qdrant_path,
        collection_name=qdrant_collection_name,
    )
    
vs = setup_qdrant()
all_text = []
all_titles = []
next_offset = None

while True:
    # 'scroll' retrieves points page-by-page without vector similarity
    records, next_offset = vs.client.scroll(
        collection_name=qdrant_collection_name,
        limit=100,           # Adjust page size as needed
        with_payload=True,   # Ensure text is included
        with_vectors=False,  # Skip vectors to save memory
        offset=next_offset
    )
    
    for record in records:
        # Extract text from the payload (default key is often 'page_content')
        text = record.payload.get("page_content", "")
        all_text.append(text)
        title = record.payload.get("metadata", "").get("title","")
        # print(title)
        all_titles.append(title)
    
    # If next_offset is None, we've reached the end
    if next_offset is None:
        break

print(f"Retrieved {len(all_text)} total documents.")
df = pd.DataFrame({'prompt': all_titles, 'completion': all_text})
df.to_excel('HTMLDOC_Extraction.xlsx',index=False)
df.to_json('HTMLDOC_Extraction.json', orient='records') #,lines=True)
vs.close()
