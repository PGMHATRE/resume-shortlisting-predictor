import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import pickle

# Load dataset
df = pd.read_csv("resume_data.csv").dropna()

# Load pre-trained BERT
bert = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for resume + JD
texts = df['resume_text'] + " " + df['job_description']
embeddings = bert.encode(texts)

# Labels
y = df['shortlisted']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Save model & BERT
pickle.dump(clf, open("model.pkl", "wb"))
pickle.dump(bert, open("bert_model.pkl", "wb"))

print("✅ Model training completed and saved.")
