# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import processing
import numpy as np
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

class Inverted_Index:
    def __init__(self, path_dictionary, path_documents, skip = 1, optimal = False):
        # skip = 1: giao (intersect)
        # skip > 1: giao + bước nhảy = skip (intersectWithSkips)
        # optimal = True: tối ưu
        self.dictionary = np.load(path_dictionary, allow_pickle=True).item()
        self.documents = np.load(path_documents, allow_pickle=True)

        if skip < 1:
            raise ValueError("skip >= 1") 
        self.skip = skip
        self.optimal = optimal

    def df(self, word):
        return len(self.dictionary[word])

    def get_posting_list(self, word):
        return self.dictionary[word]   

    def get_documents(self, listDocID):
        results = []
        for docID in listDocID: 
            results.append([docID, self.documents[docID-1]])
        return results

    def val_skip(self, p, i):
        return p[i + self.skip] 

    def hasSkip(self, p, i):
        return False if i + self.skip >= len(p) else True
    
    def intersect_2_set(self, p1, p2):
        return self.intersect(p1, p2) if self.skip == 1 else self.intersectWithSkips(p1, p2)
    
    def intersect(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j +=1       

        return answer    

    def intersectWithSkips(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                if self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                    while self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                        i += self.skip
                else:
                    i += 1
            else:
                if self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                    while self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                        j += self.skip
                else:
                    j += 1    
        return answer
    
    # tối ưu câu truy vấn
    def optimize(self, tokens): 
        # xếp theo thứ tự tăng dần df        
        return sorted(tokens, key = lambda word: self.df(word))  
    
    def query(self, str_query):

        # tách các thành phần trong câu query 
        tokens = str_query.lower().split()
        tokens = [word for word in tokens if word not in stop_words]
        
        # Thực hiện câu query      
        try:  
            if self.optimal == True:
                tokens = self.optimize(tokens)
            listDocID = self.get_posting_list(tokens.pop(0))
            while len(tokens) != 0:
                listDocID = self.intersect_2_set(listDocID , self.get_posting_list(tokens.pop(0)))
            return listDocID   
        except KeyError:
            return []

if __name__ == '__main__':


    I_index = Inverted_Index("./invertedIndex.npy", "./documents.npy", skip = 3, optimal= True)

    queries = processing.load_query()
    
    for j, query in enumerate(queries):
        print("Query", j+1)
        listDocID = I_index.query(query)
        
        for result in listDocID: 
            print(result)

        print("\n/\n")
