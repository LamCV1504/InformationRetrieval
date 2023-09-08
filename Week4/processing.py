# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

path_queries = "./query-text"

def load_query():
    with open(path_queries, 'r') as f:
        queries = f.read()

    # tách từng queryID_query
    docID_queries = queries.lower().replace("\n"," ").split("/")
    docID_queries.pop()

    # tách queryID và query
    queries = []
    for query in docID_queries:
        query = query.strip()
        index = query.find(" ")
        queries.append(query[index+1:])
    return queries

###################################################################################

if __name__ == '__main__':

    # Đọc file
    path_docs = "./doc-text-test"
    with open(path_docs, 'r') as f:
        documents = f.read()

    # tách từng docID_doc
    docID_documents = documents.lower().replace("\n"," ").split("/")
    docID_documents.pop()

    # tách docID và doc
    documents = []
    for document in docID_documents:
        document = document.strip()
        index = document.find(" ")
        documents.append(document[index+1:])

    # tạo danh sách từ 
    vocabs = list(set(" ".join(documents).strip().split()))
    vocabs = [word for word in vocabs if word not in stop_words]
    vocabs.sort(reverse=False)
    print(vocabs)

    # Tạo file từ vựng và file văn bản từ file doc - text
    np.save('./documents.npy', documents, allow_pickle=True)
    np.save('./vocabs.npy', vocabs, allow_pickle=True)

    print("Done!")

