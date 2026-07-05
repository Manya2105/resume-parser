import pdfplumber
import spacy
import os

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_entities(text):
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "DATE": [],
        "GPE": []  # locations
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

# Test on the same PDF as before
test_folder = "data/data/data/INFORMATION-TECHNOLOGY"
pdf_files = [f for f in os.listdir(test_folder) if f.endswith(".pdf")]
first_pdf = os.path.join(test_folder, pdf_files[0])

text = extract_text_from_pdf(first_pdf)
entities = extract_entities(text)

print("Extracted Entities:\n")
for label, items in entities.items():
    print(f"{label}: {list(set(items))[:10]}")  # unique, first 10