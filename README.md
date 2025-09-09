#This is a RAG model for finding incidental lung nodules in CT Chest imaging reports

##Setup:
- Clone this directory
- Currently ocr.py and rag_mistral.py will search the main directory for a subdirectoriy hardcoded as "IPN Ineligible Scans". Create this folder or modify the directory.

##Workflow:
###Run python3 ocr.py
- ocr.py will scrape text from pdfs (embedded as an image or otherwise)
- Text files will be created and pre-processed (removing headers/footers/unneeded info).

###Run python3 rag_mistral.py 
- will use regex to segment text and rule out candidates based on criteria
- Criteria: Incidental nodule found, >=6mm, no history of cancer within last 6 mos
- Will create a .json file which contains a dictrionary of criteria classifications

###Run python3 seg_eval.py
- Compares the dictionary with manually labeled data found in labels.ods and reports accuracy

##TODO:
- Move all pre-processing and filtering of text from rag_mistral.py to ocr.py or vice versa
- Implement more metrics for evaluation (precision, recall, etc)
- Manually label more docs (currently at 30)
- Currently, finding nodules and determining their size runs on two separate queries, we should reuse the chunks from the first query to improve accuracy
