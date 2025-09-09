# Requires Python 3.6 or higher due to f-strings

# Import libraries
import platform
from tempfile import TemporaryDirectory
from pathlib import Path
import glob, os
import fitz
import re

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

out_directory = """./EligibleScans/""" #os.getcwd()
out_directory = os.path.join(os.getcwd(),out_directory)

# Path of the Input pdf
PDF_files = []
for i in os.listdir(out_directory):
    if (i.endswith(".pdf") and not i.startswith('.')):
        PDF_files.append(out_directory + i)

def main():
    ''' Main execution point of the program'''
    with TemporaryDirectory() as tempdir:
        # Create a temporary directory to hold our temporary images.
        for i in range(len(PDF_files)):
            PDF_file = PDF_files[i]
            text_file = Path(PDF_file[:-3]+"txt")
            print(text_file, flush=True)

            #try extracting raw text
            doc = fitz.open(PDF_file)
            all_text = ""
            raw = False
            for page in doc:
            	all_text+= page.get_text()

            if len(all_text) >4:
                print("Raw text found!")
                #Remove irrelevant footers
                all_text = re.sub(r"""NOTE: This is information at .*?decisions.""", "", all_text,flags=re.DOTALL)
                all_text = re.sub(r"""This printed.*?only.""", "", all_text,flags=re.DOTALL)
                all_text = re.sub(r"""Medical Imaging Report""","",all_text,flags=re.DOTALL)
                all_text = re.sub(r"""Page:\s*(\d+)\s*of\s*(\d+)""","",all_text,flags=re.DOTALL)
                all_text = re.sub(r"""\*\*\*\*\* Final \*\*\*\*\*""","",all_text,flags=re.DOTALL)
                with open(text_file, "w") as output_file:
                    output_file.write(all_text)
                print(all_text)
                raw = True
            else:
                print("Raw text not found.")
            if raw == True:
            	continue
            print("Converting to image...")
            """
            Part #1 : Converting PDF to images
            """
	
            if platform.system() == "Windows":
                pdf_pages = convert_from_path(
                    PDF_file, 750, poppler_path=path_to_poppler_exe
                )
            else:
                pdf_pages = convert_from_path(PDF_file, 750)
            # Read in the PDF file at 500 DPI
            image_file_list = []
            # Iterate through all the pages stored above
            for page_enumeration, page in enumerate(pdf_pages, start=1):
                # enumerate() "counts" the pages for us.

                # Create a file name to store the image
                filename = f"{tempdir}\page_{page_enumeration:03}.jpg"

                # Declaring filename for each page of PDF as JPG

                # Save the image of the page in system
                page.save(filename, "JPEG")
                image_file_list.append(filename)

            """
            Part #2 - Recognizing text from the images using OCR
            """
            #if os.path.exists(text_file):
            #    print(str(text_file) + '\nFile Already Exists')
            #    continue
            print(text_file, flush=True)
            with open(text_file, "w", encoding="utf-8", buffering=1) as output_file:
                # Open the file in append mode so that
                # All contents of all images are added to the same file

                # Iterate from 1 to total number of pages
                text = ""
                for image_file in image_file_list:

                    # Recognize the text as string in image using pytesserct
                    text += str(((pytesseract.image_to_string(Image.open(image_file)))))

                text = text.replace("-\n", "")
               	#Remove irrelevant footers
                text = re.sub(r"""NOTE: This is information at .*?decisions.""", "", text)
                text = re.sub(r"""This printed.*?only.""", "", text)
                text = re.sub(r"""Medical Imaging Report""","",text,flags=re.DOTALL)
                text = re.sub(r"""Page:\s*(\d+)\s*of\s*(\d+)""","",text,flags=re.DOTALL)
                text = re.sub(r"""\*\*\*\*\* Final \*\*\*\*\*""","",text,flags=re.DOTALL)
                # Finally, write the processed text to the file.
                output_file.write(text)
                output_file.flush()
                os.fsync(output_file.fileno())
                output_file.close()
    
if __name__ == "__main__":
      # We only want to run this if it's directly executed!
    main()
