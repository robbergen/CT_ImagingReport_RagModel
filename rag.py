# rag_simple.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import torch, re

# 1. Load model and tokenizer (runs on GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/phi-2"

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2. Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load and split document
with open("20-0697_T0_Req_Redacted.txt", "r") as f:
    text = f.read()

def clean_ocr_text(text):
    # Remove non-ASCII chars and weird symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text = clean_ocr_text(text)


splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
docs = splitter.create_documents([text])

# 4. Embed and store in FAISS
db = FAISS.from_documents(docs, embedding_model)

# 5. Define RAG query function
def rag_query(query: str) -> str:
    retriever = db.as_retriever(search_type="similarity", k=3)
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    prompt = f"""You are a helpful assistant.

Answer the following question about a medical document. Answer briefly with any relevant info about the subject of the question. Do not add any other information or text.
    Context:
    {context}

    Question:
    {query}

    Answer:"""
    tokens = tokenizer(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 6. Run it
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_simple.py 'Your question here'")
    else:
        question = sys.argv[1]
        answer = rag_query(question)
        print("\nAnswer:\n", answer)
