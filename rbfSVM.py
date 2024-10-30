import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time


df = pd.read_csv("fake_job.csv")
print(df.shape)
print(df.columns)


df['description'] = df['description'].fillna('')  # Заповнення пропусків

tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['description'])

X = X_text
y = df['fraudulent']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()

model = SVC(kernel='rbf', gamma=0.0001)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

end_time = time.time()
execution_time = end_time - start_time
print(f"Час: {execution_time} секунд")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
