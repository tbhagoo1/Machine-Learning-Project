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
data['Text'] = data['Text'].map(lambda x: re.sub('[,.!?@$-<>]', '', x)) #Remove misc
data['Text'] = data['Text'].map(lambda x: x.lower()) #Lowercase string
data['Text']= data['Text'].map(lambda x: re.sub(r'\s+', ' ', x))  #Remove double spaces/new lines

#Turn to List and remove stop words
data.Text.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)

#Vectorize data
vectorizer = CountVectorizer(analyzer='word',lowercase=True)
data_vectorized = vectorizer.fit_transform(data)

#Get several search params and learning decays for the LDA model
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .6, .7, .8, .9]}
lda = LatentDirichletAllocation(learning_method='online', learning_offset=10.0, random_state=0)
 
#Use GridSearchCV in order to run through to get the best param
#according to Log Likelihood and and Model Perplexity
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)

#Note this uses params mentioned in
#medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6

GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
			perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [.5, .6, .7, .8, .9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

best_lda_model = model.best_estimator_

print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
