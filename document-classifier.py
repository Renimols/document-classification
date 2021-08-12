# The code below is executed in Google Colab for doing classification.
# The file reading mechanisms are implemented in similar fashion.
import io
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

from io import StringIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from wordcloud import WordCloud, STOPWORDS 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Reading data into colab env from local drive
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['test_data.csv']))
df.shape

# assigning numerical values to categories to feed into the model.
# introduced a new column named category_id
df = df[pd.notnull(df['documents'])]
df['category_id'] = df['categories'].factorize()[0]
category_id_df = df[['categories', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'categories']].values)

# Preprocessing text
lemmatizer = WordNetLemmatizer()
for i in range(0, len(df)):
    document = re.sub('[^a-zA-Z]', ' ', df['documents'][i])
    document = document.lower()
    document = document.split()
    
    document = [lemmatizer.lemmatize(word) for word in document if not word in stopwords.words('english')]
    document = ' '.join(document)
    df.documents[i] = document
  
df.head()

# Exploratory Data Analaysis
# plotting categories vs document-count:
fig = plt.figure(figsize=(8,6))
sns.catplot(x='categories', data=df, kind='count');

# plotting Document length distribution:
df['document_length'] = df['documents'].str.len()
fig = plt.figure(figsize=(8,6))
sns.distplot(df['document_length']).set_title('Document length distribution');
df['document_length'].describe()

#  box plot the df:
plt.figure(figsize=(12.8,6))
sns.boxplot(data=df, x='categories', y='document_length', width=.5);

#  remove from the 95% percentile onwards to better appreciate the histogram:
quantile_95 = df['document_length'].quantile(0.95)
print(quantile_95)
df_95 = df[df['document_length'] < quantile_95]
plt.figure(figsize=(12.8,6))
sns.distplot(df_95['document_length']).set_title('Document length distribution');

# box plot after removing outliers:
plt.figure(figsize=(12.8,6))
sns.boxplot(data=df_95, x='categories', y='document_length', width=.5);

# Feature Engineering
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.documents).toarray()
labels = df.category_id
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
  text = " ".join(unigrams[-N:]) # the input of the wordcloud generator
  wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='white', collocations=False, 
  stopwords = STOPWORDS).generate(text)
  plt.figure(figsize=(10, 8))
  plt.imshow(wordcloud) 
  plt.axis("off")
  plt.show()


# Data Modelling using Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(df['documents'], df['categories'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

test_document = "Yukos unit buyer faces loan claim The owners of embattled Russian oil giant Yukos are to ask the buyer of its former production unit to pay back a $900m (£479m) loan.State-owned Rosneft bought the Yugansk unit for $9.3bn in a sale forced by Russia to part settle a $27.5bn tax claim against Yukos. Yukos' owner Menatep Group says it will ask Rosneft to repay a loan that Yugansk had secured on its assets. Rosneft already faces a similar $540m repayment demand from foreign banks. Legal experts said Rosneft's purchase of Yugansk would include such obligations. The pledged assets are with Rosneft, so it will have to pay real money to the creditors to avoid seizure of Yugansk assets,said Moscow-based US lawyer Jamie Firestone, who is not connected to the case. Menatep Group's managing director Tim Osborne told the Reuters news agency: If they default, we will fight them where the rule of law exists under the international arbitration clauses of the credit.Rosneft officials were unavailable for comment. But the company has said it intends to take action against Menatep to recover some of the tax claims and debts owed by Yugansk. Yukos had filed for bankruptcy protection in a US court in an attempt to prevent the forced sale of its main production arm. The sale went ahead in December and Yugansk was sold to a little-known shell company which in turn was bought by Rosneft. Yukos claims its downfall was punishment for the political ambitions of its founder Mikhail Khodorkovsky and has vowed to sue any participant in the sale."
print(clf.predict(count_vect.transform([test_document])))

# Model Selection
# trying Logistic Regression,(Multinomial) Naive Bayes,Linear Support Vector Machine,Random Forest
# calculating the accuracies using cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# boxplotting the accuracies
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=4)
cv_df.groupby('model_name').accuracy.mean()

# try out explicit validation using hyperparameter tuning


# LinearSVC has highest accuracy.plotting confusion matrix 
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.categories.values, yticklabels=category_id_df.categories.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# plotting classification report of it.
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['categories'].unique()))















