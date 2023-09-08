# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Đọc file
vocabs = np.load("./vocabs.npy", allow_pickle=True)
documents = np.load("./documents.npy", allow_pickle=True)

# tạo ma trận đánh dấu
matrix = list(np.zeros((documents.shape[0], vocabs.shape[0])))

for doc in range(len(documents)):
   for word in documents[doc].split():
        if word not in stop_words:
            matrix[doc][np.where(vocabs == word)[0][0]] = 1          

# Lưu ma trận đánh dấu
np.save('./binIncidenceVectors.npy', matrix, allow_pickle=True)

print("Done!")