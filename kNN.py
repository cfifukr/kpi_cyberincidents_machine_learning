import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

for k in range(1, 21, 1):
    start_time = time.time()

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f" Кількість сусідів: {k}  /  Час: {execution_time} секунд  /  Точність: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


