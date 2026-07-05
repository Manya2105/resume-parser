import pdfplumber
import spacy
import os
import re
from skills_list import TECH_SKILLS

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_job_title(text):
    first_line = text.strip().split("\n")[0]
    return first_line.strip()

def extract_skills(text):
    text_lower = text.lower()
    found_skills = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills

def extract_dates(text):
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    return list(set(dates))

# Test on multiple PDFs to check consistency
test_folder = "data/data/data/INFORMATION-TECHNOLOGY"
pdf_files = [f for f in os.listdir(test_folder) if f.endswith(".pdf")][:5]  # first 5

for pdf_file in pdf_files:
    pdf_path = os.path.join(test_folder, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    
    print(f"\n--- {pdf_file} ---")
    print("Job Title:", extract_job_title(text))
    print("Skills found:", extract_skills(text))