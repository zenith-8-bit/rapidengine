import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() if page.extract_text() else ""
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return text
def extract_text_from_docx(doc_path):
    text = ""
    try: 
        doc = Document('your_doc.docx'); 
        for para in doc.paragraphs:
            text = text + para.text
    except FileNotFoundError:
        print('error file is not there')
    except Exception as e:
        print(e)
    return text
    

# --- Example Usage ---
# Create a dummy PDF file for demonstration if you don't have one
# In a real scenario, you would replace 'example.pdf' with your actual file path.
# For this example, we'll write a simple text file and pretend it's a PDF for testing
# (Note: PyPDF2 expects a proper PDF, this is just to show the path usage)
# If you have a PDF, you can upload it to Colab and use its path.

# To make this runnable without a real PDF, let's create a placeholder function
# However, for actual PDF extraction, you need a valid PDF file.

# Uncomment and modify the following line with the path to your PDF file:
pdf_file_path = "/content/tata_capital_annual_report.pdf"
extracted_pdf_text = extract_text_from_pdf(pdf_file_path)
if extracted_pdf_text:
    print("\n--- Extracted Text from PDF ---")
    print(extracted_pdf_text[:500]) # Print first 500 characters
else:
    print("No text extracted from PDF.")

