import numpy as np
import pandas as pd
import sklearn
import nltk
import re
import json

import matplotlib.pyplot as plt
import pylab

""" Работа с Датасетом
    1. Предоставить ссылку
    2. Записать сам датасет
    3. Вывести 5 первых строк
"""

dataset_url = "https://raw.githubusercontent.com/IBM/nlc-email-phishing/master/data/Email-testingdata.json"
dataset = pd.read_json(dataset_url)
print(dataset.head())

"""Создание графика отношения спама к обычным сообщениям"""
pylab.subplot(1,2,1)
pylab.rcParams["figure.figsize"] = [8,10]
dataset.Class.value_counts().plot(kind='pie', autopct='%1.0f%%')
pylab.title("Круговая диаграмма")

"""На данном этапе мы находим число обычных сообщений, а также сколько в них слов"""
dataset_ham = dataset[dataset['Class'] == "ham"]
dataset_ham_count = dataset_ham['Text'].str.split().str.len()
dataset_ham_count.index = dataset_ham_count.index.astype(str) + ' words:'
dataset_ham_count.sort_index(inplace=True)

"""Повторим это для спама"""
dataset_spam = dataset[dataset['Class'] == "spam"]
dataset_spam_count = dataset_spam['Text'].str.split().str.len()
dataset_spam_count.index = dataset_spam_count.index.astype(str) + ' words:'
dataset_spam_count.sort_index(inplace=True)

"""Построим гистограмму, в которой мы отношение числа слов на число сообщений разного класса"""
bins = np.linspace(0, 50, 10)
pylab.subplot(1,2,2)
pylab.hist([dataset_ham_count, dataset_spam_count], bins, label=['ham', 'spam'])
pylab.legend(loc='upper right')
pylab.title("Гистограмма значений")

pylab.show()

"""Подготовка данных: 
Нам требуется удалить все специальные символы и цифры, а также все пробелы, которые образуются из-за удаления"""

def text_preprocess(sen):
    sen = re.sub('[^a-zA-Z]', ' ', sen)
    sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen)
    sen = re.sub(r'\s+', ' ', sen)
    return sen

"""Создадим метки по тексту и классу"""
X = dataset["Text"]
y = dataset["Class"]

"""Проведем препроцессинг на нашем датасете"""
X_messages = []
messages = list(X)
for mes in messages:
    X_messages.append(text_preprocess(mes))

"""Произведем конвертацию чисел в вектора, а также удалим английские стоп-слова"""
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X= tfidf_vec.fit_transform(X_messages).toarray()

"""Разделение данных на обучающие и тестовые наборы"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)

"""Обучение алгоритмом машинного обучения 'Логистическая регрессия'"""
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)

"""Оценка алгоритма"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

