import numpy as np
from scipy.sparse import csr_matrix as cm
from numpy.linalg import norm
from numpy.linalg import svd
from numpy import power

def load_data(path):
    data = np.loadtxt(path)
    return data

if __name__ == "__main__":
    data = load_data('rates/ratings_train_1.txt')
    # omega = cm((data[:,2], (data[:,0].astype(np.int32), data[:,1].astype(np.int32))))
    # print("data length: %d" % len(data))
    # print(omega.getnnz())
    # adj = load_data('adjs/adj_UNB.res')
    # obj = cm((adj[:,2], (adj[:,0].astype(np.int32), adj[:,1].astype(np.int32))))
    # print('adj length: %d' % len(adj))
    # print(obj.getnnz())
    data_uni = set([(i[0],i[1]) for i in data])
    print(len(data_uni))