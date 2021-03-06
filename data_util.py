#coding=utf8
'''
    utils to process data for experiments
'''
import time
import logging

import numpy as np
import pickle

class DataLoader(object):
    '''
        load the train and test data, including the representations for users and items, generated by meta-graph
        input: the given filenames of train and test data
        return: train_X, train_Y, test_X, test_Y.
        Besides, can print the information of the data
    '''
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir')
        self.train_filename = config.get('train_filename')
        self.test_filename = config.get('test_filename')
        self.item_num = len(set(np.loadtxt('yelp_dataset/rates/ratings_train_1.txt')[:, 1]))
        self.N = config.get('N')
        self.F = config.get('F')
        self.L = config.get('L')
        if config.get('dt') == 'synthetic':
            self._load_random_data()
        elif config.get('file_type') == 'pickle':
            self._load_pickle()
        else:
            self._load()

    def _load_random_data(self):
        self.train_X = np.loadtxt(self.data_dir + self.config.get('train_X'))
        self.train_Y = np.loadtxt(self.data_dir + self.config.get('train_Y'))
        self.test_X = np.loadtxt(self.data_dir + self.config.get('test_X'))
        self.test_Y = np.loadtxt(self.data_dir + self.config.get('test_Y'))

    def _load(self):
        # Needs modify to fit pickle file
        start_time = time.time()

        train_data = np.loadtxt(self.data_dir + self.train_filename)
        test_data = np.loadtxt(self.data_dir + self.test_filename)
        train_num = train_data.shape[0]
        test_num = test_data.shape[0]

        uid2reps, bid2reps = self._load_representation()

        self.train_X = np.zeros((train_num, self.N))
        self.train_Y = train_data[:,2]
        self.test_X = np.zeros((test_num, self.N))
        self.test_Y = test_data[:,2]

        ind = 0
        for u, b, _ in train_data:
            ur = uid2reps[int(u)]   # int(u) for uid, has a len(metapath)*10 vector
            br = bid2reps[int(b)]   # int(b) for bid
            self.train_X[ind] = np.concatenate((ur,br)) # concatenate (embed_u, embed_b), so each item in train_X is a long vector
            ind += 1
        X_sparsity = np.count_nonzero(self.train_X) * 1.0 / self.train_X.size

        ind = 0
        for u, b, _ in test_data:
            ur = uid2reps.get(int(u), np.zeros(self.N//2))
            br = bid2reps.get(int(b), np.zeros(self.N//2))
            self.test_X[ind] = np.concatenate((ur,br))
            ind += 1

        test_X_sparsity = np.count_nonzero(self.test_X) * 1.0 / self.test_X.size

    def _generate_feature_files(self):
        meta_graphs = self.config.get('meta_graphs')
        topK = self.config.get('topK')
        ufiles, vfiles = [], []
        for graph in meta_graphs:
            if graph == 'ratings_only':
                ufiles.append('ratings_only_user.dat')
                vfiles.append('ratings_only_item.dat')
            else:
                ufiles.append('%s_user.dat' % (graph))
                vfiles.append('%s_item.dat' % (graph))
        return ufiles, vfiles

    def _load_representation(self):
        '''
            load user and item latent features generate by MF for every meta-graph
        '''
        #if dt in ['yelp-200k', 'amazon-200k', 'amazon-50k', 'amazon-100k', 'amazon-10k', 'amazon-5k', 'cikm-yelp', 'yelp-50k', 'yelp-10k', 'yelp-5k', 'yelp-100k', 'douban']:
        fnum = self.N // 2
        ufilename = self.data_dir + 'adjs/ind2uid'
        bfilename = self.data_dir + 'adjs/ind2bid'
        with open(ufilename, 'rb') as f:
            uids = pickle.load(f).keys()
        with open(bfilename, 'rb') as f:
            bids = pickle.load(f).keys()
        # uids = [int(l.strip()) for l in open(ufilename, 'r').readlines()]
        uid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in uids}

        # bids = [int(l.strip()) for l in open(bfilename, 'r').readlines()]   # read in bids.txt
        bid2reps = {k:np.zeros(fnum, dtype=np.float64) for k in bids}       # build a dict for each bid

        ufiles, vfiles = self._generate_feature_files() # ufiles: files of users, vfiles: files of items
        # ratings or top500 of each metapath
        # 10-dim vector for each id(uid or bid)

        feature_dir = self.data_dir + self.config.get('feature_dir')
        for find, filename in enumerate(ufiles):
            ufs = np.loadtxt(feature_dir + filename, dtype=np.float64)
            cur = find * self.F # self.F: rank for MF to generate latent features, self.F = 10
            for uf in ufs:
                uid = int(uf[0])
                f = uf[1:]  # from the second number to the last number in each line, 10-dim vector
                uid2reps[uid][cur:cur+self.F] = f   # each uid2reps element has a len(ufiles)*10

        for find, filename in enumerate(vfiles):
            bfs = np.loadtxt(feature_dir + filename, dtype=np.float64)
            cur = find * self.F
            for bf in bfs:
                bid = int(bf[0])
                f = bf[1:]
                bid2reps[bid][cur:cur+self.F] = f
        logging.info('load all representations, len(ufiles)=%s, len(vfiles)=%s, ufiles=%s, vfiles=%s', len(ufiles), len(vfiles), '|'.join(ufiles), '|'.join(vfiles))
        return uid2reps, bid2reps

    def get_exp_data(self):
        return self.train_X, self.train_Y, self.test_X, self.test_Y, self.item_num

if __name__ == "__main__":
    item_num = len(set(np.loadtxt('yelp_dataset/rates/ratings_train_1.txt')[:, 1]))
    print(item_num)

