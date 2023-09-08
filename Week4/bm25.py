# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np
import math

from nltk.tokenize import word_tokenize
import re

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
rank = 5

def load_data(path):
    with open(path, 'r') as f:
        data = f.read()
    ID_data = data.lower().replace("\n"," ").split("/")
    ID_data.pop()
    data = []
    for datum in ID_data:
        datum = datum.strip()
        index = datum.find(" ")
        data.append(datum[index+1:]) 
    return data

def tokenize_and_remove_stopwords(doc):
    tokens = word_tokenize(re.sub(r'[^\w\s]', '', doc).lower())
    filtered_tokens = filter(lambda token: token not in stop_words, tokens)
    return list(filtered_tokens)


class BM25:

    def __init__(self, k1=2, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):

        tf = []
        df = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))
                   
            # tính tf trong mỗi document
            freq = {}
            for term in document:
                term_count = freq.get(term, 0) + 1
                freq[term] = term_count
            tf.append(freq)
            

            # tính df cho mỗi term
            for term, _ in freq.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size 
        self.tf_ = tf
        self.df_ = df
        return self

    def search(self, query):
        scores = [self._sorce(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator =  math.log10(self.corpus_size_/self.df_[term]) * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)
             
        return score

if __name__ == "__main__":

    # tokenzie, loại bỏ stop-words
    documents = load_data("./doc-text")
    list_words_docs = list(map(lambda doc: tokenize_and_remove_stopwords(doc), documents))

    bm25 = BM25()
    bm25.fit(list_words_docs)
    
    # Phần truy vấn
    queries = load_data("./query-text")
    list_words_queries = list(map(lambda doc: tokenize_and_remove_stopwords(doc), queries))

    for i in range(len(list_words_queries)):
        scores = bm25.search(list_words_queries[i])
        scores_index = np.argsort(scores)
        scores_index = scores_index[::-1]

        print("Query", i+1, ":", queries[i],"\n")
        print("Ranking top ",rank)
        for i in range(rank):
            print("Doc", scores_index[i]+1, "- Score:", scores[scores_index[i]])
            print(documents[scores_index[i]])
        print("\n/\n")

    