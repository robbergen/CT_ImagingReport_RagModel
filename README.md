# RAG model for finding incidental lung nodules in CT Chest imaging reports

## Setup:
- Clone this directory
- Currently ocr.py and rag_mistral.py will search for a subdirectory hardcoded as "IPN Ineligible Scans". This is where your data (.pdf, .txt) should go. (This will be updated in the future)
- Place Mistral 7b-instruct model in a 'models' folder

## Workflow:
### Run python3 ocr.py
- ocr.py will scrape text from pdfs (embedded as an image or otherwise)
- Text files will be created and pre-processed (removing headers/footers/unneeded info).

### Run python3 rag_mistral.py 
- will use regex to segment text and rule out candidates based on criteria
- Criteria: Incidental nodule found, >=6mm, no history of cancer within last 6 mos
- Will create a out_dict.json file which contains a dictionary of criteria classifications

### Run python3 seg_eval.py
- Compares the dictionary with manually labeled data (N=30) found in labels.ods and reports accuracy

## TODO:
- Move all pre-processing and filtering of text from rag_mistral.py to ocr.py or vice versa (right now it's scattered everywhere)
- Implement more metrics for evaluation (precision, recall, etc)
- Manually label more docs (currently at 30)
- Include optional user input for input directories
- Try updating to a medical RAG (https://github.com/AquibPy/Medical-RAG-LLM)
- Currently, finding nodules and determining their size runs on two separate queries, we should reuse the chunks from the first query to improve accuracy
