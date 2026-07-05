import pdfplumber
import os
import re
from skills_list import TECH_SKILLS

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_skills(text):
    text_lower = text.lower()
    found_skills = []
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills

def match_resume_to_jd(resume_skills, jd_skills):
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    
    matched = resume_set & jd_set
    missing = jd_set - resume_set
    extra = resume_set - jd_set
    
    match_score = len(matched) / len(jd_set) * 100 if jd_set else 0
    
    return {
        "match_score": round(match_score, 1),
        "matched_skills": list(matched),
        "missing_skills": list(missing),
        "extra_skills": list(extra)
    }

# Test with a sample JD
sample_jd = """
We are looking for an IT Manager with strong experience in network administration.
Required skills: Cisco, Active Directory, VMware, Linux, Project Management, SQL.
Experience with Citrix and budgeting is a plus.
"""

jd_skills = extract_skills(sample_jd)
print("Skills extracted from JD:", jd_skills)

# Test against a real resume
test_folder = "data/data/data/INFORMATION-TECHNOLOGY"
pdf_files = [f for f in os.listdir(test_folder) if f.endswith(".pdf")]
test_pdf = os.path.join(test_folder, pdf_files[0])

resume_text = extract_text_from_pdf(test_pdf)
resume_skills = extract_skills(resume_text)
print("\nSkills extracted from resume:", resume_skills)

result = match_resume_to_jd(resume_skills, jd_skills)

print("\n" + "=" * 50)
print("JD MATCH RESULTS")
print("=" * 50)
print(f"Match Score: {result['match_score']}%")
print(f"Matched Skills: {result['matched_skills']}")
print(f"Missing Skills: {result['missing_skills']}")
print(f"Extra Skills (resume has, JD doesn't need): {result['extra_skills']}")