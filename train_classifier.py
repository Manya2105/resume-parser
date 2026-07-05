import pdfplumber
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Load data from all 4 categories
categories = ["INFORMATION-TECHNOLOGY", "ENGINEERING", "BUSINESS-DEVELOPMENT", "FINANCE"]
base_path = "data/data/data"

texts = []
labels = []
# Data augmentation: add modern SWE/DevOps resume examples to IT category
# (training data was skewed toward traditional IT-ops/sysadmin skills)
synthetic_it_resumes = [
    """Software Engineer Python JavaScript React Node.js Django FastAPI Docker Kubernetes AWS Jenkins CI/CD
    PostgreSQL MongoDB Redis Git GitHub REST API microservices system design agile""",
    
    """Full Stack Developer TypeScript React Express.js Node.js MongoDB PostgreSQL Docker AWS Lambda
    Jenkins GitHub Actions Jest pytest CI/CD pipeline REST API GraphQL agile scrum""",
    
    """Backend Engineer Python Java Django FastAPI AWS Docker Kubernetes PostgreSQL MongoDB Redis
    Prometheus observability CI/CD Jenkins Git REST API microservices distributed systems""",
    
    """DevOps Engineer Docker Kubernetes AWS Jenkins CI/CD Terraform Prometheus Git Linux
    Python Bash scripting cloud infrastructure automation monitoring deployment""",
    
    """Machine Learning Engineer Python TensorFlow PyTorch scikit-learn AWS SageMaker Docker
    Git REST API FastAPI PostgreSQL pandas numpy deep learning NLP computer vision""",
    
    """Software Development Engineer Python Java JavaScript AWS Docker React Node.js PostgreSQL
    MongoDB Redis Git Jenkins CI/CD agile REST API system design microservices""",
    
    """Cloud Engineer AWS Azure Docker Kubernetes Terraform Jenkins CI/CD Python Bash Linux
    Git monitoring Prometheus PostgreSQL REST API infrastructure automation""",
    
    """Data Engineer Python SQL PostgreSQL MongoDB AWS Spark pandas Git Docker REST API
    ETL pipeline data warehousing airflow kafka data modeling""",
    
    """Frontend Engineer JavaScript TypeScript React Vue.js Node.js CSS HTML Git Docker AWS
    Jest CI/CD REST API GraphQL agile scrum responsive design""",
    
    """Site Reliability Engineer Python Bash Docker Kubernetes AWS Prometheus Jenkins Git Linux
    CI/CD monitoring alerting infrastructure automation PostgreSQL Redis""",
    
    """Software Engineer intern Python JavaScript React AWS Docker PostgreSQL Git REST API
    agile CI/CD Node.js Django FastAPI MongoDB redis microservices""",
    
    """Junior Developer Python Java JavaScript React Node.js PostgreSQL MongoDB Git Docker AWS
    REST API agile scrum CI/CD Jenkins HTML CSS""",
    
    """AI Engineer Python TensorFlow PyTorch NLP deep learning AWS Docker FastAPI Git
    scikit-learn pandas numpy REST API PostgreSQL MLOps model deployment""",
    
    """Systems Engineer Python Java AWS Docker Kubernetes Jenkins CI/CD PostgreSQL MongoDB
    Git Linux REST API microservices distributed systems agile""",
    
    """Software Architect Python Java JavaScript AWS Docker Kubernetes PostgreSQL MongoDB Redis
    Git Jenkins CI/CD REST API GraphQL microservices system design agile""",
    
    """Research Engineer Python PyTorch TensorFlow scikit-learn AWS Docker Git PostgreSQL
    pandas numpy deep learning NLP computer vision REST API FastAPI""",
    
    """Mobile Developer Python JavaScript React Native Node.js AWS MongoDB PostgreSQL Git Docker
    REST API agile CI/CD Jenkins Firebase""",
    
    """Security Engineer Python AWS Docker Kubernetes Jenkins CI/CD Git Linux PostgreSQL
    REST API penetration testing vulnerability assessment monitoring""",
    
    """Platform Engineer Python Bash AWS Docker Kubernetes Terraform Jenkins CI/CD Git Linux
    Prometheus monitoring PostgreSQL Redis infrastructure automation""",
    
    """Graduate Software Engineer Python JavaScript React Node.js PostgreSQL MongoDB Git Docker
    AWS REST API agile CI/CD Jenkins HTML CSS machine learning"""
]

for resume_text in synthetic_it_resumes:
    texts.append(resume_text)
    labels.append("INFORMATION-TECHNOLOGY")

print("Loading resumes...")
for category in categories:
    folder = os.path.join(base_path, category)
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                texts.append(text)
                labels.append(category)
        except Exception as e:
            print(f"Skipped {pdf_file}: {e}")

print(f"Loaded {len(texts)} resumes total\n")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train set: {len(X_train)} resumes")
print(f"Test set: {len(X_test)} resumes\n")

# Vectorize
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("CLASSIFIER RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer for later use in API
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved as classifier.pkl and vectorizer.pkl")