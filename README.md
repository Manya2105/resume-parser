# Resume Parser & JD Matcher

An end-to-end NLP + ML system that parses resume PDFs, extracts structured information, classifies job categories, and scores resume-to-job-description fit using both keyword matching and semantic similarity.

## Live Demo
> Deploy link coming soon

## Features

- **PDF Text Extraction** — Parses unstructured resume PDFs using pdfplumber
- **Skill Extraction** — Custom 70+ term taxonomy spanning technical, business, finance, and sales domains
- **Job Category Classifier** — TF-IDF + Logistic Regression model trained on 495 resumes across 4 categories (88.89% accuracy, 80/20 train-test split)
- **JD Keyword Matcher** — Set-based skill overlap scoring between resume and job description
- **Semantic Similarity** — SBERT (all-MiniLM-L6-v2) sentence embeddings with cosine similarity for contextual matching beyond exact keywords
- **ATS Score** — Weighted composite score (keyword match 40% + semantic similarity 40% + skills depth 20%)
- **REST API** — FastAPI backend with `/parse-resume` and `/match-jd` endpoints
- **Frontend UI** — Dark-themed single-page interface with drag-and-drop upload

## Tech Stack

- **NLP:** spaCy, Sentence-Transformers (SBERT)
- **ML:** scikit-learn (TF-IDF, Logistic Regression)
- **Backend:** FastAPI, Uvicorn
- **PDF Parsing:** pdfplumber
- **Frontend:** HTML, CSS, JavaScript

## ML Pipeline

### Job Category Classifier
- Dataset: 495 resumes across 4 categories (IT, Engineering, Business Development, Finance)
- Features: TF-IDF vectorization (2000 features, English stop words removed)
- Model: Logistic Regression (max_iter=1000)
- Split: 80/20 train-test, stratified
- Results: 88.89% accuracy, precision/recall above 0.88 across all classes

### Semantic JD Matching
- Model: `all-MiniLM-L6-v2` (90MB, 384-dim embeddings)
- Method: Cosine similarity between resume and JD sentence embeddings
- Handles terminology variations (e.g. "ML" vs "Machine Learning") that exact keyword matching misses

## Known Limitations

- Classifier trained on traditional professional resumes — lower confidence on modern SWE/DevOps profiles (training data skew identified and documented)
- Skill extraction uses keyword matching — incidental mentions (e.g. "customer service" in project description) may be flagged as skills
- Name extraction uses heuristics (first line, word count, case) — may fail on non-standard resume formats

## Dataset

Training data: [Resume Dataset by snehaanbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle) — not included in repo due to size.

## Setup

```bash
git clone https://github.com/Manya2105/resume-parser.git
cd resume-parser
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_classifier.py  # trains and saves model files
uvicorn main:app --reload
```

Open `frontend.html` in your browser.

## API Endpoints

`POST /parse-resume` — Upload a PDF, returns name, job title, skills, predicted category, confidence

`POST /match-jd` — Upload a PDF + paste JD text, returns keyword match score, semantic similarity, ATS score, matched/missing skills