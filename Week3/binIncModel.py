# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

class BinIndependenceModel:
    def __init__(self, vectors, vocabs, documents):
        self.vectors = vectors
        self.feature_names = vocabs
        self.documents = documents
        self.weights = None
    
    def get_document(self, docID):
        return self.documents[docID-1]
    
    # get vector của 1 doc
    def get_vector(self, docID):
        return self.vectors[docID-1]
    
    # get các feature_names
    def get_feature_names_out(self):
        return self.feature_names
    
    # get Matrix các doc vector
    def get_vectors(self):
        return self.vectors      
    
    # khởi tạo weights ban đầu
    def init_RSV_weights(self):
        N = self.documents.shape[0]
        n = np.sum(self.vectors, axis = 0) #  n = df
        # Ta có: ci = log((pi.(1 - ri))/(ri.(1 - pi))) 
        # Giả sử pi = 0.5 (Croft and Harper (1979)) với mọi xi và S << N (số văn bản phù hợp rất thấp)
        # => ci = log((1 - ri)/ri) và ri = (ni - s) / (N - S) ~ ni / N
        # => ci = log((1 - ni/N)/(ni/N)) = log((N - ni)/ni) ~ log((N - ni + 0.5)/ (ni + 0.5)) tránh Infinity
        self.weights = np.log10((N - n + 0.5) / (n + 0.5)) # ~ idf ~ N / (ni + 0.5)
    
    # Xếp hạng
    def ranking(self, vectorQuery):
        indicies = np.where(vectorQuery==1) # xi_Q = 1
        # Tính điểm RSVd = Tổng<xi = qi = 1>(ci)   
        scores = np.sum(self.vectors[:, *indicies] * self.weights[indicies].T, axis = 1)
        # trả về scores của các docs theo thứ tự giảm dần    
        ranks = np.argsort(-scores)
        return ranks, scores[ranks]
    
    # tính trọng số weight lại khi có phản hồi                                  
    def recompute_weights(self, relevant_doc, vectorQuery):
        # N: Số văn bản
        N = self.documents.shape[0] 
        # N_rel = |V| số văn bản xếp hạng cao nhất (phù hợp)
        N_rel = relevant_doc.shape[0] 
        # index of qi = 1
        qi_1 = np.where(vectorQuery == 1) 
        # Số văn bản có chứa xi trong tập V
        n_vi = np.sum(self.vectors[relevant_doc][:,*qi_1], axis = 0) 
        # pi = (|Vi| + 0.5) / (|V| + 1)
        pi = (n_vi + 0.5) /( N_rel + 1) 
        n = np.sum(self.vectors[:, *qi_1], axis = 0) # df
        # ri = (n - |Vi| + 0.5) / (N - |Vi| + 1)
        ri = (n - n_vi + 0.5) / (N - N_rel + 1) 
        # Cập nhật lại weight: ci = log((pi.(1-ri))/(ri.(1-pi))) 
        self.weights[qi_1] = np.log10((pi/(1-pi))/(ri/(1-ri))) 

    def answer_query(self, query): 
        # Tạo vector cho câu query
        # Không cần loại bỏ stops - words vì features đã được xử lý trước k chứa stop - words 
        vectorQuery = np.isin(self.feature_names, query.split()).astype(int) 
        # Phù hợp phản hồi giả lặp
        # khởi tạo pi = 0.5 ^ S << N => ri ~ ni / N 
        self.init_RSV_weights()
        # Thực hiện đến khi nào hội tụ 
        # Tức đến khi giá trị của pi và ri không thay đổi nhiều (d(c_new,c_old) < epsilon)  
        # hoặc đã lặp đến số lần tối đa 
        N_rel = 5 # Số văn bản xếp hạng cao nhất cho cải thiện xếp hạng 
        n_loop = 10 # Số vòng lặp tối đa
        epsilon = 0.00001 # ngưỡng thiết lặp cho d(c_new,c_old)
        while True:
            n_loop = n_loop - 1
            ranks, _ = self.ranking(vectorQuery)
            weights_old = np.array(self.weights)
            self.recompute_weights(ranks[:N_rel], vectorQuery) # VR = V
            if n_loop == 0 or np.sqrt(np.sum((self.weights - weights_old)**2, axis = 0)) < epsilon:
                return self.ranking(vectorQuery)
    
if __name__ == "__main__":

    bim = BinIndependenceModel(
        np.load("./binIncidenceVectors.npy", allow_pickle=True),
        np.load("./vocabs.npy", allow_pickle=True), 
        np.load("./documents.npy", allow_pickle=True)
    )
    
    # Biểu diễn các doc dưới Tập các vector nhị phân
    import pandas as pd
    df = pd.DataFrame(
        dtype = int,
        data = bim.get_vectors(),
        columns = bim.get_feature_names_out(),
        index = [f"Doc {i+1}" for i in range(len(bim.documents))]
    )

    print("\n==================================================================\n")
    
    from processing import load_query
    queries = load_query()
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}\n")                
        ranking, scores = bim.answer_query(query)
        for index, score in list(zip(ranking, scores))[:5]:
            print(f"\nDocument {index+1}: {bim.get_document(index+1)} \nScore: {score} \n/")
        print("/")
