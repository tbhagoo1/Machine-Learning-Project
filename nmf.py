import os
import pandas as pd
import re

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

BASE_PATH = "archive"
DATASET_TYPE = "food"
NUM_TOPICS = 9

def get_files_to_read(directory_name):

    file_list = os.listdir(directory_name)
    return file_list

def read_file(base_path, file_name, dataset_type):
    """Returns df of specified review text from dataset

    Args:
        base_path (str): Base directory of file, ex: "archive"
        file_name (str): File name: "food.csv" or "reviews.csv"
        dataset_type (str): Specifies either food or amazon products

    Returns:
        [pandas df]: Dataframe of review text
    """

    file_path = os.path.join(base_path, file_name)
    file_data = pd.read_csv(file_path)    

    if dataset_type == "food":
        review_data = file_data[['Text']].copy()
    else:
        review_data = file_data[['reviews_text']].copy()
    
    # add .sample(100) for testing
    print(review_data.head())

    return review_data

def prepare_text_regex(text_df, dataset_type):
    """Taxes text and applies regex filtering for words

    Args:
        text_df (dataframe): The dataframe of review text
        dataset_type (str): The type of dataset: "food" or "product"

    Returns:
        text_df (dataframe): Filtered email dataframe after regex
    """

    if dataset_type == "food":
        column_name = "Text"
    else:
        column_name = "reviews_text"

    text_df[column_name] = \
    text_df[column_name].map(lambda x: re.sub('[,\.!?]', '', x))
    text_df[column_name] = \
    text_df[column_name].map(lambda x: re.sub('www', '', x))
    text_df[column_name] = \
    text_df[column_name].map(lambda x: x.lower())
    text_df[column_name].head()

    return text_df

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stopwords):
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stopwords] for doc in texts]

def get_NMF_topics(model, vectorizer, top_word_num, num_topics):
    """Processes data using NMF model

    Args:
        model (NMF): NMF model class
        vectorizer (Vectorizer class): sklearn Vectorizer class
        top_word_num (int): Number of words to get for each topic
        num_topics (int): Number of topics to look for in dataset

    Returns:
        nmf_df (dataframe): Pandas dataframe containing topics and top words
    """
    feature_names = vectorizer.get_feature_names()
    top_words_dict = {}
    for i in range(num_topics):
        word_ids = model.components_[i].argsort()[:-top_word_num - 1:-1]
        words = [feature_names[key] for key in word_ids]
        words = [re.sub('\s+', ' ', word) for word in words]
        words = [re.sub("\'", "", word) for word in words]
        top_words_dict[f'Topic #{i+1}'] = words

    nmf_df = pd.DataFrame(top_words_dict)
    # nmf_df.to_csv("nmf_topics.csv")
    
    return nmf_df

def main():

    # text_files = get_files_to_read(BASE_PATH)

    df = read_file(BASE_PATH, "food.csv", "food")
    processed_df = prepare_text_regex(df, "food")

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'br'])

    # This is to differentiate the two datasets: amazon products and food reviews
    # since they have different column names for the text data
    if DATASET_TYPE == "food":
        data = processed_df["Text"].values.tolist()
    else:
        data = processed_df["reviews_text"].values.tolist()

    data_words = list(sent_to_words(data))# remove stop words
    data_words = remove_stopwords(data_words, stop_words)
    # print("printing datawords")
    # print(data_words[0])

    import gensim.corpora as corpora# Create Dictionary
    id2word = corpora.Dictionary(data_words)# Create Corpus
    # print("printing id2word")
    # print(id2word)
    texts = data_words# Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]# View
    # print("printing corpus")
    # print(corpus[0])
    # print(len(corpus))
    

    # Can use this to test word count!
    # for i in range(len(corpus[0])):
    #     print(f"Word {corpus[0][i][0]} [{id2word[corpus[0][i][0]]}] -- count: {corpus[0][i][1]} times")


    sentences = [' '.join(text) for text in data_words]
    # print("printing articles sentences")
    # print(sentences)

    vectorizer = CountVectorizer(analyzer='word', max_features=2000)
    x = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer()
    x_tfid = transformer.fit_transform(x)

    x_tfid_norm = normalize(x_tfid, norm='l1', axis=1)

    num_topics = NUM_TOPICS
    nmf_model = NMF(n_components=num_topics, init="nndsvd")
    nmf_model.fit(x_tfid_norm)

    get_NMF_topics(nmf_model, vectorizer, 20, num_topics)


if __name__ == "__main__":
    main()
