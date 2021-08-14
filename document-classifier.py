# The code below is executed in Google Colab for doing classification.
# Each process could be executed in different blocks for better clarity

from wordcloud import WordCloud, STOPWORDS
from google.colab import files
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from io import StringIO
import io
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')


# Reading data into colab env from local drive

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['test_data.csv']))
df.shape

# assigning numerical values to categories to feed into the model.
# introduced a new column named category_id

df = df[pd.notnull(df['documents'])]
df['category_id'] = df['categories'].factorize()[0]
category_id_df = df[['categories', 'category_id']
                    ].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'categories']].values)

# Exploratory Data Analaysis
# plotting categories vs document-count:

fig = plt.figure(figsize=(8, 6))
sns.catplot(x='categories', data=df, kind='count')

# plotting Document length distribution:

df['document_length'] = df['documents'].str.len()
fig = plt.figure(figsize=(8, 6))
sns.distplot(df['document_length']).set_title('Document length distribution')
df['document_length'].describe()

#  box plot the df:

plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df, x='categories', y='document_length', width=.5)

#  remove from the 95% percentile onwards to better appreciate the histogram:

quantile_95 = df['document_length'].quantile(0.95)
print(quantile_95)
df_95 = df[df['document_length'] < quantile_95]
plt.figure(figsize=(12.8, 6))
sns.distplot(df_95['document_length']).set_title(
    'Document length distribution')

# box plot after removing outliers:

plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df_95, x='categories', y='document_length', width=.5)

# Preprocessing text

df_95=df_95.reset_index() 
lemmatizer = WordNetLemmatizer()
for i in range(0, len(df_95)):
    document = re.sub('[^a-zA-Z]', ' ', df_95['documents'][i])
    document = document.lower()
    document = document.split()

    document = [lemmatizer.lemmatize(
        word) for word in document if not word in stopwords.words('english')]
    document = ' '.join(document)
    df_95.documents[i] = document

df_95.head()

# Feature Engineering

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df_95.documents).toarray()
labels = df_95.category_id
features.shape

# using sklearn.feature_selection.chi2 to find the terms(unigrams) that are the most correlated with each of the products:

N = 10
for categories, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    print("# '{}':".format(categories))
    print("  . Most correlated unigrams:\n. {}".format(unigrams[-N:]))
    text = " ".join(unigrams[-N:])  # the input of the wordcloud generator
    wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', collocations=False,
                          stopwords=STOPWORDS).generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

# Model Selection
# trying Logistic Regression,(Multinomial) Naive Bayes,Linear Support Vector Machine,Random Forest
# calculating the accuracies using cross_val_score with default hyperparams

# split input data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, random_state=0)
models = [
    RandomForestClassifier(),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(
        model, X_train, y_train, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# boxplotting the accuracies:

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=4)
cv_df.groupby('model_name').accuracy.mean()

# finding best hyperparameters using RandomizedSearchCV

penalty = ['l2', 'l1']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid param_distributions
grid = dict(penalty=penalty, C=c_values)
lsvc = LinearSVC()
rf_random = RandomizedSearchCV(estimator=lsvc, param_distributions=grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
rf_random.best_params_

# calculated the accuracy mean of cross_val_score with best hyper parameters.
# but default values were the best

# even though LinearSVC() has the highest cross_val_score,
# training set accuracy and test set accuracy is calculated for all models.

models.append(LinearSVC(C=100))
for model in models:
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # plotting confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.categories.values, yticklabels=category_id_df.categories.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # plotting classification report of the model [test set accuracy].
    print(metrics.classification_report(y_test, y_pred,
          target_names=df_95['categories'].unique()))
    # plotting classification report of the model [traing set accuracy].
    print(metrics.classification_report(y_train, y_pred_train,
          target_names=df_95['categories'].unique()))

# observed LinearSVC() has highest performance with test data set also.
# LinearSVC() with rf_random.best_params_has the best accuracy
# building the selected model using best hyperparameters and test with sample input
# {0: 'entertainment', 1: 'tech', 2: 'business', 3: 'politics', 4: 'sport'}

uploaded = files.upload()
input = uploaded['sample.txt'].decode("utf-8").split("\r\n")
preprocessed_input = []
document = re.sub('[^a-zA-Z]', ' ', input[0])
document = document.lower()
document = document.split()
document = [lemmatizer.lemmatize(
    word) for word in document if not word in stopwords.words('english')]
document = ' '.join(document)
preprocessed_input.append(document)
classifier = LinearSVC(C=100).fit(X_train, y_train)
category_id = classifier.predict(tfidf.transform(preprocessed_input))[0]
print("This document belongs to Category-", id_to_category[category_id])
