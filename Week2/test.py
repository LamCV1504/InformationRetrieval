# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np
import pandas as pd
from main import tokenize_and_remove_stopwords, computeTFIDF, cosin_similarity, transform_vector_TFIDF

documents = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system respone time", "The ESP user interface management system", "System and human system engineering testing of ESP", "Relation of user perceived respone time to error measurement", "The generation of random binary unordered trees", "The intersection graph of paths in trees", "Graph minors IV widths of trees and well quasi ordering", "Graph minors a survey"]
query = "The intersection of graph survey and trees"

if __name__ == "__main__":

    # tokenzie, loại bỏ stop-words
    list_words_docs = list(map(lambda doc: tokenize_and_remove_stopwords(doc), documents))

    feature_names, idf, matrix_tfidf = computeTFIDF(list_words_docs)

    list_words_queries = tokenize_and_remove_stopwords(query)

    vector_query = transform_vector_TFIDF(list_words_queries, feature_names, idf)

    scores, indicies = cosin_similarity(np.array(vector_query), np.array(matrix_tfidf))

    index = ["Query"]
    index.extend(["Doc " + str(i+1) for i in range(len(documents))])

    print("\n TF-IDF VALUES")
    df = pd.DataFrame(
        np.concatenate((vector_query, matrix_tfidf), axis=0), 
        columns = feature_names, 
        index = index
    )
    print(df)

    print("\n RANKING")
    df = pd.DataFrame(scores, columns = ["Cosine"], index = index[1:])
    df["Rank"] = [list(indicies).index(i) + 1 for i in range(len(documents))]
    print(df)
