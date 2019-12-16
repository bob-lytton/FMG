#-*-coding:utf-8-*-
import json
import numpy as np
from copy import deepcopy
import pickle
import time
from scipy.sparse import csr_matrix as csr

def load_jsondata_from_file(path, ftype=None):
    """
    return data are index data\\
    filtering result should be combined with the original json files
    """
    print("loading %s" % path)
    t0 = time.time()
    data = []
    with open(path, 'r') as f:
        if ftype == None:
            for line in f:
                item = json.loads(line)
                data.append(item)
        elif ftype == 'user':
            for line in f:
                item = json.loads(line)
                data.append({'user_id': item['user_id'], 'friends': item['friends']})
        elif ftype == 'business':
            for line in f:
                item = json.loads(line)
                data.append({'business_id': item['business_id'], 'categories': item['categories'], 'city': item['city']})
        elif ftype == 'review':
            for line in f:
                item = json.loads(line)
                data.append({'user_id': item['user_id'], 'business_id': item['business_id'], 'stars': item['stars']})
    print("loading %s done, time cost %.2f" % (path, time.time()-t0))
    return data

def load_pickle(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    return data

def get_id_to_ind(json_datas, filtered_list, filtered_name, id_name, multi_value):
    """
    first appear position
    id for data_id, ind for index
    the same as set() and dict()
    """
    ind2id = {}
    id2ind = {}
    tot = 0
    for data in json_datas:
        if data[filtered_name] not in filtered_list:
            continue
        if multi_value:
            # this get a list of data_id(str) with no spaces in the beginning or end
            data_ids = [data_id.strip() for data_id in data[id_name].split(',')] # strip to delete the spaces in the beginning or end of a line
        else:
            data_ids = [data[id_name]]
        for data_id in data_ids:
            if data_id not in id2ind:
                ind2id[tot] = data_id    # input ind, output id
                id2ind[data_id] = tot    # input id, output ind
                tot = tot + 1
    return ind2id, id2ind

def divide_rate(review_train):
    """
    divide 1~5 rates to pos and neg classes
    <= 3: neg, >= 4: pos
    """
    pos_reviews = []
    neg_reviews = []
    for review in review_train:
        if review['rate'] > 3.0:
            pos_reviews.append([review['user_id'], review['business_id'], review['rate']])
        elif review['rate'] <= 3.0:
            neg_reviews.append([review['user_id'], review['business_id'], review['rate']])
    return pos_reviews, neg_reviews

def dataset_split(reviews, uid2ind, bid2ind, train_ratio, valid_ratio, test_ratio):
    """
    split the dataset as train, valid, and test set through train_ratio, valid_ratio, and test_ratio
    """
    selected_reviews = []
    for review in reviews:
        if (review['user_id'] not in uid2ind) or (review['business_id'] not in bid2ind):
            continue
        filtered_review = {}
        filtered_review['user_id'] = uid2ind[review['user_id']]
        filtered_review['business_id'] = bid2ind[review['business_id']]
        filtered_review['rate'] = int(review['stars'])
        selected_reviews.append(filtered_review)

    n_reviews = len(selected_reviews)   # indber of selected reviews
    test_indices = np.random.choice(range(n_reviews), size=int(n_reviews*test_ratio), replace=False) # randomly choose some indices as test set

    left = set(range(n_reviews)) - set(test_indices)
    n_left = len(left)

    valid_indices = np.random.choice(list(left), size=int(n_left*valid_ratio), replace=False)
    train_indices = list(left - set(valid_indices))

    train_data = [selected_reviews[index] for index in train_indices]
    valid_data = [selected_reviews[index] for index in valid_indices]
    test_data = [selected_reviews[index] for index in test_indices]
    return train_data, valid_data, test_data

def get_adj_matrix(uid2ind, bid2ind, city_id2ind, cat_id2ind, users, businesses, pos_reviews, neg_reviews):
    """
    metapaths: UPB, UNB, UUPB, UUNB, UPBUB, UNBUB, UPBCaB, UNBCaB, UPBCiB, UNBCiB
    metapaths about pos are all pos, about neg are all neg
    """
    tot_users = len(uid2ind)  # tot for total
    tot_business = len(bid2ind)
    tot_city = len(city_id2ind)
    tot_category = len(cat_id2ind)
    #relation U-U
    adj_UU = np.zeros([tot_users, tot_users])
    adj_UPB = np.zeros([tot_users, tot_business])
    adj_UNB = np.zeros([tot_users, tot_business])
    adj_BCa = np.zeros([tot_business, tot_category])
    adj_BCi = np.zeros([tot_business, tot_city])
    for user in users:
        if user['user_id'] not in uid2ind:
            continue
        user_id = uid2ind[user['user_id']]
        for friend in user['friends'].split(','):
            friend = friend.strip()
            if friend in uid2ind:
                friend_id = uid2ind[friend]
                adj_UU[user_id][friend_id] = 1
                adj_UU[friend_id][user_id] = 1
    #relation U-P-B
    for review in pos_reviews:
        user_id = review[0]
        business_id = review[1]
        adj_UPB[user_id][business_id] = 1
    #relation U-N-B
    for review in neg_reviews:
        user_id = review[0]
        business_id = review[1]
        adj_UNB[user_id][business_id] = 1
    #relation B_Ca B_Ci
    for business in businesses:
        if business['business_id'] not in bid2ind:
            continue
        business_id = bid2ind[business['business_id']]
        city_id = city_id2ind[business['city']]
        adj_BCi[business_id][city_id] = 1
        for category in business['categories'].split(','):
            category = category.strip()
            category_id = cat_id2ind[category]
            adj_BCa[business_id][category_id] = 1

    #metapath
    adj_UUPB = adj_UU.dot(adj_UPB)
    adj_UUNB = adj_UU.dot(adj_UNB)

    adj_UPBU = adj_UPB.dot(adj_UPB.T)
    adj_UNBU = adj_UNB.dot(adj_UNB.T)

    adj_UPBUB = adj_UPBU.dot(adj_UPB)
    adj_UNBUB = adj_UNBU.dot(adj_UNB)

    adj_UPBCa = adj_UPB.dot(adj_BCa)
    adj_UPBCaB = adj_UPBCa.dot(adj_BCa.T)

    adj_UNBCa = adj_UNB.dot(adj_BCa)
    adj_UNBCaB = adj_UNBCa.dot(adj_BCa.T)

    adj_UPBCi = adj_UPB.dot(adj_BCi)
    adj_UPBCiB = adj_UPBCi.dot(adj_BCi.T)

    adj_UNBCi = adj_UNB.dot(adj_BCi)
    adj_UNBCiB = adj_UNBCi.dot(adj_BCi.T)

    return adj_UPB, adj_UNB, adj_UUPB, adj_UUNB, adj_UPBUB, adj_UNBUB, adj_UPBCaB, adj_UNBCaB, adj_UPBCiB, adj_UNBCiB


if __name__ == "__main__":
    filtered_user = load_pickle('filtered/users.pickle')
    filtered_business = load_pickle('filtered/businesses.pickle')
    filtered_reviews = load_pickle('filtered/reviews.pickle')   # success!

    user_json     = load_jsondata_from_file('json/yelp_academic_dataset_user.json') # 25s   60.5s new
    business_json = load_jsondata_from_file('json/yelp_academic_dataset_business.json') # 4.45s   7.62s new
    review_json   = load_jsondata_from_file('json/yelp_academic_dataset_review.json')   # 69.8s    237.67s new 

    ind2uid, uid2ind         = get_id_to_ind(user_json, filtered_user, 'user_id', 'user_id', False)
    ind2bid, bid2ind         = get_id_to_ind(business_json, filtered_business, 'business_id', 'business_id', False)
    ind2city_id, city_id2ind = get_id_to_ind(business_json, filtered_business, 'business_id', 'city', False)
    ind2cat_id, cat_id2ind   = get_id_to_ind(business_json, filtered_business, 'business_id', 'categories', True)
    print("user_id2ind: %s" % len(uid2ind))
    print("business_id2ind: %s" % len(bid2ind))
    print("city_id2ind: %s" % len(city_id2ind))
    print("category_id2ind: %s" % len(cat_id2ind))

    r = (ind2uid, ind2bid, ind2city_id, ind2cat_id)
    r_names = ('ind2uid', 'ind2bid', 'ind2city_id', 'ind2cat_id')

    for i in range(len(r)):
        with open('adjs/' + r_names[i], 'wb') as f:
            pickle.dump(r[i], f, protocol=4)

    # dataset split
    review_train, review_valid, review_test = dataset_split(filtered_reviews, uid2ind, bid2ind, 0.8, 0.1, 0.2)
    # in review_train: {'rate':number, 'user_id':number, 'business_id':number}
    # train valid test data save
    print("generating ratings dataset")
    d = (review_train, review_valid, review_test)
    d_names = ('ratings_train_1', 'ratings_valid_1', 'ratings_test_1')
    for i in range(len(d)):
        with open('rates/' + d_names[i] + '.txt', 'w') as f:
            for item in d[i]:
                f.write(str(item['user_id'])+' '+str(item['business_id'])+' '+str(item['rate'])+'\n')

    # train data divide rate
    pos_reviews, neg_reviews = divide_rate(review_train)

    # cal adjacent matrices
    adj_UPB, adj_UNB, adj_UUPB, adj_UUNB, adj_UPBUB, adj_UNBUB, adj_UPBCaB, adj_UNBCaB, adj_UPBCiB, adj_UNBCiB = \
        get_adj_matrix(uid2ind, bid2ind, city_id2ind, cat_id2ind, user_json, business_json, pos_reviews, neg_reviews)

    # relation save
    t = (adj_UPB, adj_UNB, adj_UUPB, adj_UUNB, adj_UPBUB, adj_UNBUB, adj_UPBCaB, adj_UNBCaB, adj_UPBCiB, adj_UNBCiB)
    t_names = ('adj_UPB', 'adj_UNB', 'adj_UUPB', 'adj_UUNB', 'adj_UPBUB', 'adj_UNBUB', 'adj_UPBCaB', 'adj_UNBCaB', 'adj_UPBCiB', 'adj_UNBCiB')
    for i in range(len(t)):
        with open('adjs/' + t_names[i] + '.res', 'w') as f:
            for uid, line in enumerate(t[i]):
                for bid, num in enumerate(line):
                    if num != 0:
                        write_str = '%d %d %.1f\n' % (uid, bid, num)
                        f.write(write_str)

