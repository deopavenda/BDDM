
# 🌐 Install dependensi
!pip install vaderSentiment scikit-learn joblib pandas nltk

# 📚 Import
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk, joblib
nltk.download('stopwords')
from nltk.corpus import stopwords

# 📤 Upload file CSV
from google.colab import files
uploaded = files.upload()

# 📄 Baca data
df = pd.read_csv('reddit_wsb.csv')
df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

# ✨ VADER Sentiment Score
sid = SentimentIntensityAnalyzer()
df['vader_score'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# 🏷️ Label otomatis: Compound > 0 → Positif, <= 0 → Negatif
df['label'] = (df['vader_score'] > 0).astype(int)

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# ⚙️ Pipeline: TF-IDF + RandomForest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 🚀 Training
pipeline.fit(X_train, y_train)

# 📊 Evaluasi
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 💾 Simpan model
joblib.dump(pipeline, 'sentiment_model.pkl')
files.download('sentiment_model.pkl')
