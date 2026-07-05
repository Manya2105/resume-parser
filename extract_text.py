import pdfplumber
import os

# Path to one category folder to test
test_folder = "data/data/data/INFORMATION-TECHNOLOGY"

# Get the first PDF in that folder
pdf_files = [f for f in os.listdir(test_folder) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs in {test_folder}")

# Test extraction on the first one
first_pdf = os.path.join(test_folder, pdf_files[0])
print(f"\nExtracting text from: {first_pdf}\n")

with pdfplumber.open(first_pdf) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

print(text[:1000])  # print first 1000 characters to check it worked