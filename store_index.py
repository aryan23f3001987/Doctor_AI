# from dotenv import load_dotenv
# import os
# from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
# from pinecone import Pinecone
# from pinecone import ServerlessSpec 
# from langchain_pinecone import PineconeVectorStore

# load_dotenv()


# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
# OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
# os.environ["OPENAI_PROJECT_ID"] = OPENAI_PROJECT_ID or ""


# extracted_data=load_pdf_file(data='data/')
# filter_data = filter_to_minimal_docs(extracted_data)
# text_chunks=text_split(filter_data)

# embeddings = download_hugging_face_embeddings()

# pinecone_api_key = PINECONE_API_KEY
# pc = Pinecone(api_key=pinecone_api_key)



# index_name = "medical-chatbot"  # change if desired

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# index = pc.Index(index_name)


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings, 
# )

#-----------------------------------------------------------------------------------------------------------------
# store_index.py

from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# 👇 Apna helper functions import kar
from src.helper import load_pdf_file, filter_to_minimal_docs  

# --------------------------
# 1. Load env variables
# --------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
os.environ["OPENAI_PROJECT_ID"] = OPENAI_PROJECT_ID or ""

# --------------------------
# 2. Load + Split PDFs
# --------------------------
print("📂 Loading PDF files...")
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)

print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # smaller chunks → safe for Pinecone
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
text_chunks = splitter.split_documents(filter_data)

print(f"✅ Total chunks: {len(text_chunks)}")

# --------------------------
# 3. OpenAI Embeddings
# --------------------------
print("🧠 Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # size = 1536
    api_key=OPENAI_API_KEY
)

# --------------------------
# 4. Pinecone Setup
# --------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print("🪄 Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # must match embedding model dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# --------------------------
# 5. Upload in Batches
# --------------------------
print("⬆️ Uploading chunks to Pinecone in batches...")

batch_size = 100
for i in tqdm(range(0, len(text_chunks), batch_size)):
    batch = text_chunks[i : i + batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        index_name=index_name,
        embedding=embeddings,
    )

print("🎉 All chunks uploaded successfully to Pinecone!")