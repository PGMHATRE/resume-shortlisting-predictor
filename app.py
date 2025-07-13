import streamlit as st
import pdfplumber
import pickle
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Load model & BERT
model = pickle.load(open("model.pkl", "rb"))
bert = SentenceTransformer('all-MiniLM-L6-v2')

# Sample skill list
skill_keywords = [
    "python", "java", "sql", "c++", "machine learning", "data analysis", "pandas",
    "numpy", "power bi", "tableau", "streamlit", "html", "css", "javascript",
    "flask", "django", "excel", "tensorflow"
]

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Extract skills from text
def extract_skills(text):
    text = text.lower()
    return [skill for skill in skill_keywords if skill in text]

# Streamlit UI
st.set_page_config(page_title="Resume Shortlisting Predictor", layout="centered")
st.title("📄 Resume Shortlisting Predictor")

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
jd_input_type = st.radio("📑 Provide Job Description", ["Upload PDF", "Paste Text"])
job_text = ""

if jd_input_type == "Upload PDF":
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    if jd_file:
        job_text = extract_text_from_pdf(jd_file)
else:
    job_text = st.text_area("Paste Job Description here")



def generate_pdf(prediction, confidence, matched, missing, match_percent):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    c.drawString(100, 750, "Resume Shortlisting Report")
    c.line(100, 745, 400, 745)

    c.drawString(100, 720, f"Prediction: {'Shortlisted ✅' if prediction == 1 else 'Not Shortlisted ❌'}")
    c.drawString(100, 700, f"Confidence Score: {round(confidence * 100, 2)}%")
    c.drawString(100, 680, f"Skill Match Score: {match_percent}%")

    c.drawString(100, 650, "Matched Skills:")
    for i, skill in enumerate(matched):
        c.drawString(120, 630 - i*15, f"- {skill}")

    c.drawString(100, 590 - len(matched)*15, "Missing Skills:")
    for j, skill in enumerate(missing):
        c.drawString(120, 570 - len(matched)*15 - j*15, f"- {skill}")

    c.save()
    buffer.seek(0)
    return buffer


# ✅ Single button to handle everything
if st.button("🚀 Predict + Analyze Skills"):
    if not resume_file or not job_text.strip():
        st.warning("Please upload both resume and job description.")
    else:
        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)

        # Combine & predict
        combined_text = resume_text + " " + job_text
        embedding = bert.encode([combined_text])
        prediction = model.predict(embedding)[0]
        prob = model.predict_proba(embedding)[0][1]

        st.success(f"Prediction: {'✅ Shortlisted' if prediction == 1 else '❌ Not Shortlisted'}")
        st.info(f"Confidence Score: {round(prob * 100, 2)}%")

        # 🔍 Skill analysis
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_text)

        matched = list(set(resume_skills) & set(jd_skills))
        missing = list(set(jd_skills) - set(resume_skills))
        match_percent = round(len(matched) / len(jd_skills) * 100, 2) if jd_skills else 0

        st.subheader("🧠 Skill Match Results")
        st.markdown(f"✅ **Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")
        st.info(f"🔢 Skill Match Score: {match_percent}%")

        # 📊 Pie Chart
        labels = ['Matched Skills', 'Missing Skills']
        values = [len(matched), len(missing)]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        fig.update_traces(marker=dict(colors=['#28a745', '#dc3545']))
        fig.update_layout(title="📊 Skill Match Breakdown")
        st.plotly_chart(fig, use_container_width=True)
        # # 📊 Pie Chart
        # labels = ['Matched Skills', 'Missing Skills']
        # values = [len(matched), len(missing)]

        # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        # fig.update_traces(marker=dict(colors=['#28a745', '#dc3545']))
        # fig.update_layout(title="📊 Skill Match Breakdown")
        # st.plotly_chart(fig, use_container_width=True)

        # 📄 PDF Report Download
        pdf_buffer = generate_pdf(prediction, prob, matched, missing, match_percent)

        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_buffer,
            file_name="resume_report.pdf",
            mime="application/pdf"
        )


                # 🌥️ Word Cloud of Resume
        st.subheader("☁️ Resume Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_text)

        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        # 💡 Improvement Tips
        st.subheader("💡 Improvement Suggestions")
        if missing:
            st.markdown("To increase your shortlisting chances, consider adding these skills or tools to your resume (if you have experience):")
            for skill in missing:
                st.markdown(f"- ✏️ Add **{skill}** (if relevant)")
        else:
            st.markdown("✅ Your resume aligns well with the job description. No major skills missing!")






