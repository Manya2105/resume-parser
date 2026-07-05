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

# Run across multiple categories
categories = ["INFORMATION-TECHNOLOGY", "ENGINEERING", "BUSINESS-DEVELOPMENT", "FINANCE"]
base_path = "data/data/data"

all_results = []
failed = 0

for category in categories:
    folder = os.path.join(base_path, category)
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    print(f"Processing {len(pdf_files)} resumes from {category}...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                failed += 1
                continue
            skills = extract_skills(text)
            all_results.append({
                "file": pdf_file,
                "category": category,
                "job_title": extract_job_title(text),
                "num_skills": len(skills),
                "skills": skills
            })
        except Exception as e:
            print(f"Failed on {pdf_file}: {e}")
            failed += 1

# Calculate stats
total = len(all_results)
with_skills = sum(1 for r in all_results if r["num_skills"] > 0)
avg_skills = sum(r["num_skills"] for r in all_results) / total if total > 0 else 0

all_skills = [skill for r in all_results for skill in r["skills"]]
from collections import Counter
top_skills = Counter(all_skills).most_common(10)

print("\n" + "=" * 50)
print(f"BENCHMARK RESULTS (across {len(categories)} categories)")
print("=" * 50)
print(f"Total resumes processed: {total}")
print(f"Failed extractions: {failed}")
print(f"Resumes with at least 1 skill detected: {with_skills} ({with_skills/total*100:.1f}%)")
print(f"Average skills per resume: {avg_skills:.2f}")
print(f"\nTop 10 most common skills:")
for skill, count in top_skills:
    print(f"  {skill}: {count}")

# Breakdown by category
print(f"\nBreakdown by category:")
for category in categories:
    cat_results = [r for r in all_results if r["category"] == category]
    cat_avg = sum(r["num_skills"] for r in cat_results) / len(cat_results) if cat_results else 0
    print(f"  {category}: {len(cat_results)} resumes, avg {cat_avg:.2f} skills")