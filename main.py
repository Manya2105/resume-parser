from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import re
import shutil
import os
import pickle
from skills_list import TECH_SKILLS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_job_title(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    first_line = lines[0] if lines else ""
    word_count = len(first_line.split())
    is_likely_name = word_count <= 4 and (first_line.istitle() or first_line.isupper())
    if is_likely_name and len(lines) > 1:
        return {"name": first_line, "job_title": lines[1]}
    else:
        return {"name": None, "job_title": first_line}

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

def semantic_match(resume_text, jd_text):
    resume_embedding = sbert_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = sbert_model.encode(jd_text, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding)
    return round(float(similarity[0][0]) * 100, 1)

def calculate_ats_score(keyword_match, semantic_similarity, num_skills):
    skills_bonus = min(num_skills / 15 * 100, 100)
    ats_score = (keyword_match * 0.4) + (semantic_similarity * 0.4) + (skills_bonus * 0.2)
    if ats_score >= 70:
        verdict = "Strong Match"
    elif ats_score >= 45:
        verdict = "Moderate Match"
    else:
        verdict = "Weak Match"
    return round(ats_score, 1), verdict

def predict_category(text):
    vec = vectorizer.transform([text])
    prediction = clf.predict(vec)[0]
    probabilities = clf.predict_proba(vec)[0]
    confidence = max(probabilities) * 100
    return prediction, round(confidence, 1)

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text_from_pdf(temp_path)
    title_info = extract_job_title(text)
    skills = extract_skills(text)
    category, confidence = predict_category(text)
    os.remove(temp_path)
    return {
        "name": title_info["name"],
        "job_title": title_info["job_title"],
        "skills": skills,
        "num_skills": len(skills),
        "predicted_category": category,
        "confidence": confidence
    }

@app.post("/match-jd")
async def match_jd(file: UploadFile = File(...), jd_text: str = Form(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    resume_text = extract_text_from_pdf(temp_path)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    os.remove(temp_path)
    result = match_resume_to_jd(resume_skills, jd_skills)
    result["resume_skills"] = resume_skills
    result["jd_skills"] = jd_skills
    result["semantic_similarity"] = semantic_match(resume_text, jd_text)
    ats_score, verdict = calculate_ats_score(
        result["match_score"],
        result["semantic_similarity"],
        len(resume_skills)
    )
    result["ats_score"] = ats_score
    result["verdict"] = verdict
    return result

@app.get("/")
async def root():
    return {"message": "Resume Parser API is running"}