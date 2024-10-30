import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv("fake_job.csv")
print(df.shape)
print(df.columns)


df['description'] = df['description'].fillna('')

tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['description'])

X = X_text
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
