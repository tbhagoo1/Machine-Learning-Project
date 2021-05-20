import numpy as np 
import pandas as pd
import re
import spacy
import gensim
from gensim.utils import simple_preprocess
import nltk
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

#functions
def stopwords():
	stopwordList =[]
	with open("stopwords.txt",'r') as File:
		for line in File:
			for word in line.split():
				stopwordList.append(word.lower())
		
	return stopwordList
	
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]
             
#Tokenize
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
	


#Data
data = pd.read_csv('FoodReviews.csv',low_memory = False)# Print head
stop_words = stopwords()

#Clean Data
data = data.loc[data['Text'].str.len() > 60]
data.Text.values.tolist()
data = [re.sub('[,\.!?@$-<>]', '', x) for x in data] #Remove nonalphanumerics
data = [re.sub(r'\s+',' ', x) for x in data] #Remove double spaces/new lines

#Tokenize and remove stop words and stem
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)

#Vectorize data
vectorizer = CountVectorizer(analyzer='word',lowercase=True,max_features=500000)
data_vectorized = vectorizer.fit_transform(data)

search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
lda = LatentDirichletAllocation(learning_method='online', learning_offset=10.0,
  max_iter=10, random_state=0)
  
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)

GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
			perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

best_lda_model = model.best_estimator_

print("Best Model's Params: ", model.best_params_)

print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
