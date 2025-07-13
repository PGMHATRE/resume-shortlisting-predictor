# 📄 Resume Shortlisting Predictor

A smart AI-powered web app that predicts whether a resume matches a job description using BERT embeddings and machine learning. It analyzes skills, visualizes results with charts and word clouds, and even generates a downloadable PDF report for users.

🚀 [Live Demo](https://resume-shortlisting-predictor.streamlit.app/) • 📁 [Download Project](https://github.com/PGMHATRE/resume-shortlisting-predictor)

---

## ✨ Features

- 🔍 **Resume & Job Description Matching** (PDF or text input)
- 🤖 **ML-Based Shortlisting Prediction** using BERT + Logistic Regression
- 📊 **Skill Match Score** with matched/missing skill analysis
- ☁️ **Word Cloud** visualization of resume content
- 📄 **Downloadable PDF Report** of results
- ✅ Deployed via Streamlit Cloud – ready for recruiters to try!

---

## 🧠 How It Works

1. Upload a **resume (PDF)** and a **job description** (PDF or paste).
2. The app uses **Sentence-BERT embeddings** to compute semantic similarity.
3. A trained **Logistic Regression** model predicts the shortlisting outcome.
4. It extracts key skills, compares them, and shows:
   - ✅ Matched skills
   - ❌ Missing skills
   - 📊 Pie chart + ☁️ Word cloud
5. Download a professional **PDF report** of your result!

---

## 🛠️ Tech Stack

| Component            | Tools Used                          |
|----------------------|--------------------------------------|
| Frontend             | Streamlit                           |
| Machine Learning     | Scikit-learn, Sentence-Transformers |
| Text Parsing         | pdfplumber, WordCloud, regex        |
| Charts & Visuals     | Plotly, Matplotlib                  |
| PDF Report           | reportlab                           |
| Deployment           | Streamlit Cloud                     |



## 📦 Installation

```bash
git clone https://github.com/your-username/resume-shortlisting-predictor.git
cd resume-shortlisting-predictor
pip install -r requirements.txt
streamlit run app.py
