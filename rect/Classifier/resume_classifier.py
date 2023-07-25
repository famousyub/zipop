from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn import metrics
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import string
import re
import spacy
import en_core_web_sm
import pickle
import pandas as pd
import matplotlib.pyplot as plt

punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()

# Custom transformer using spaCy
class Predictors(TransformerMixin):
    def transform(self, X, **transform_params):

        # Cleaning Text
        cleaned_text = [clean_text(text) for text in X]

        # complete_text = ''
        # for text in cleaned_text:
        #   complete_text += text
        # makeWordCloud(complete_text)

        return cleaned_text

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Debugger
class Debug(TransformerMixin):

    def transform(self, X):
        print(pd.DataFrame(X))
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self



# Basic function to clean the text
def clean_text(text):

    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\n')
    text = text.replace('\\r', '\n')
    # text = text[2:-1] #remove 'b
    text = text.replace("'b", ' ')
    text = re.sub('nan ', ' ', text)
    text = re.sub(r'\\x[0-9a-z]{2}', r' ', text)
    text = re.sub(r'[0-9]{2,}', r' ', text)
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', ' ', text)  # remove hashtags
    text = re.sub('@\S+', ' ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text.lower()
    text = re.sub(r'xx+', r' ', text)
    text = re.sub(r'XX+', r' ', text)

    return text.strip()

def makeWordCloud(data):
    wc = WordCloud().generate(data)
    plt.figure(figsize=(15, 15))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Creating our tokenizer function
def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    # -PRON- is for personal pronouns
    mytokens = [word.lemma_.lower().strip() if word.lemma_ !=
                "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [
        word for word in mytokens if word not in stop_words and word not in punctuations]

    return mytokens


# Vectorizer
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1), max_features=3500)
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=3500)


def train(resumeDataSet):

    X = resumeDataSet['Resume']
    ylabels = resumeDataSet['Category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, ylabels, test_size=0.3)

    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    # Create pipeline
    pipe = Pipeline([('cleaner', Predictors()),
                    ('vectorizer', tfidf_vector),
                    ('classifier', classifier)])

    # Model Generation
    pipe.fit(X_train, y_train)

    # Predicting with a test dataset
    predicted = pipe.predict(X_test)

    # Model Accuracy
    print("Classification Report:",
        metrics.classification_report(y_test, predicted))

    with open('Models/localmodel2.pickle', 'wb') as f:
        pickle.dump(pipe, f)


def classify(testing_data):

    # print("Test Data: ", testing_data)
    # testing_data = pd.read_json(testing_data)
    pickle_in = open('Models\localmodel2.pickle', 'rb')
    pipe = pickle.load(pickle_in)
    predictions = pipe.predict(testing_data)
    return predictions


if __name__ == '__main__':

    resumeDataSet = pd.read_csv(
        'Data/resume_eda_linkedin2.csv', encoding='utf-8')
    resumeDataSet = resumeDataSet[['Category', 'Resume']]

    # train(resumeDataSet)

    testing_data = pd.read_csv('Data/resume_pdf_trimmed.csv', encoding='utf-8')
    testing_data = testing_data.Resume
    print(classify(testing_data))
