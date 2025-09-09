from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain

import re, glob,os
import json

# 1. Initialize local llama-cpp-python model with your .gguf file
model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=4092,
    n_threads=8,
    # Adjust n_gpu_layers based on GPU VRAM 
    n_gpu_layers=128,
    verbose=False,
)

# 2. Initialize embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load and clean document
def clean_ocr_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

rootpath = os.path.join(os.getcwd(),'IPN Ineligible Scans')
txt_files = [os.path.join(rootpath,i) for i in os.listdir(rootpath) if i.endswith(".txt")]
txt_files.sort()
txt_dict = {}
num_docs = len(txt_files)

#Grab patient identifier from the file name
def ident(filename: str) -> str:
	pattern = re.compile(r'.*/(?:20-)?(\d{4})')
	return pattern.match(filename).group(1)

def split_and_store(text):
	splitter = RecursiveCharacterTextSplitter(chunk_size=32, chunk_overlap=16)
	docs = splitter.split_text(text)
	docs = [Document(page_content=chunk) for chunk in docs]

	#Build FAISS vector store
	db = FAISS.from_documents(docs, embedding_model)
	return db

#Define rag query
def rag_query(db, query: str, max_tokens: int = 256) -> str:
	retriever = db.as_retriever(search_type="similarity", k=8)
	context_docs = retriever.invoke(query)
	context = "\n".join([doc.page_content for doc in context_docs])

	mapped_outputs = []
	mapped_docs = []
	for doc in context_docs:
		context = doc
		prompt = f"""### Instruction:
Using the following context, answer the question concisely.

Context:
{context}

Question:
{query}

### Response:"""

		response = llm(prompt, max_tokens=256, stop=["### Response:"])
		response = response['choices'][0]['text'].strip()
		mapped_outputs.append(response)
		mapped_docs.append(doc)
		#print('doc: ', doc)
		#print('response', response)

	#pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|millimet(?:er|re)s?)\b",re.IGNORECASE)
	return mapped_outputs, context_docs

def regexhistory_query(doc):
	pattern = re.compile(r'(?i)history:(.*?)(?=[A-Z]+:|\Z)')
	match = pattern.search(doc)
	if match:
		return match.group(1).strip()
	else:
		return "NA"

def regexreason_query(doc):
	pattern = re.compile(r'(?i)reason[A-Za-z\s]*?:(.*?)(?=[A-Z]+:|\Z)')
	match = pattern.search(doc)
	if match:
		return match.group(1).strip()
	else:
		return "NA"

def regexcomparison_query(doc):
	pattern = re.compile(r'(?i)comparison:(.*?)(?=[A-Z]+:|\Z)')
	match = pattern.search(doc)
	if match:
		return match.group(1).strip()
	else:
		return "NA"

def regexfindings_query(doc):
	pattern = re.compile(r'FINDINGS:\s*[\s\S]*',re.IGNORECASE)
	match = pattern.search(doc)
	if match:
		doc = match.group(0).strip()
	pattern2 = re.compile(r'(?is)Procedure\s+Description\s*:?.*?Modality\s*:?\s*\S+')
	pattern3 = re.compile(r'\(\d+\s*[/:]\s*\d+\)')
	doc = pattern2.sub('',doc)
	doc = pattern3.sub('',doc)
	return doc

def nodule_query(db):
	question = """Determine whether or not this document contains a nodule, pulmonary nodule, lung nodule, lobe nodule, lingula nodule, ground glass nodule, or subpleural nodule. Do not include any other information or explanations. Return either TRUE or FALSE.
"""

	answer,context_docs = rag_query(db, question)
	has_true = any([s.lower() == "true" for s in answer])
	#print(answer)

	bool_answer = [s.lower() == "true" for s in answer]
	query = """This document contains information about the size of a nodule, pulmonary nodules, lung nodule, lobe nodule, lingula nodule, ground glass nodule, or subpleural nodule. 
Return the size of the largest nodule. You must return a number with a unit of measurement in mm or cm. If no number is given return 0. DO NOT EXPLAIN YOUR REASONING.
"""
	for i,doc in enumerate(context_docs):
		context = doc
		prompt = f"""### Instruction:
Using the following context, answer the question concisely.

Context:
{context}

Question:
{query}

### Response:"""

		response = llm(prompt, max_tokens=256, stop=["### Response:"])
		response = response['choices'][0]['text'].strip()
		if bool_answer[i]:
			print('context', context)
			print('response', response)

	return has_true

def nodule_size(db):
	question = """This document contains information about the size of a nodule, nodular mass, nodular density, pulmonary nodules, lung nodules, lobe nodules, lingula nodules, ground glass nodules, or subpleural nodules. 
Return the size of the largest nodule converting to units of millimeters if necessary. Return only the largest number, do not explain anything. If no number is given return 0.
"""

	answer, context_docs = rag_query(db, question)
	#Clean the answer
	#print(answer)
	for s in answer:
		numbers = []

		for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(cm|mm)?",s, flags=re.IGNORECASE):
			value = float(match.group(1))
			unit = match.group(2)
			if unit:
				unit = unit.lower()
				if unit == "cm":
					value *=10
			numbers.append(value)
	if numbers:
		return max(numbers) >= 6.0
	else:
		return False
	#print(answer)
	#return answer

#Output dict
out_dict = {}

for i in range(num_docs):
	with open(txt_files[i], "r", encoding="utf-8") as f:
    		text = f.read()
	id = ident(txt_files[i])
	#text = clean_ocr_text(text)
	txt_dict[i] = text
	f.close()
	#print("File:\n",txt_files[i])

	regexhistory = regexhistory_query(clean_ocr_text(text)) #Stupid workaround for now, cleaning breaks the findings regex
	regexreason = regexreason_query(clean_ocr_text(text))
	regexfindings = regexfindings_query(text)
	regexcomparison = regexcomparison_query(clean_ocr_text(text))

	#Change db to findings only to help nodule queries
	db = split_and_store(regexfindings)
	nodule = nodule_query(db)
	nodulesize = nodule_size(db)
	#print(nodule, nodulesize)
	out_dict[id] = {'history':regexhistory,'reason':regexreason,'findings':regexfindings,
'comparison':regexcomparison, 'nodule_present':nodule, 'nodule_size':nodulesize}

	#nodule_size = nodule_size_query()
	#print(nodule_size)

with open("out_dict.json", "w") as f:
	json.dump(out_dict, f, indent=4)

	#Final eval here

# 7. Run CLI interface
#if __name__ == "__main__":
#    import sys
#    if len(sys.argv) < 2:
#        print("Usage: python rag_mistral.py 'Your question here'")
#        sys.exit(1)
#    question = sys.argv[1]
#    answer = rag_query(question)
#    print("\nAnswer:\n", answer)
