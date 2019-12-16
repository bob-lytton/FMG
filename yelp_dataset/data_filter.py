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

def filter_rare_node(users, businesses, reviews, user_thresh, business_thresh, friend_thresh):
    """
    filter the nodes with few activities (reviews in this example)
    input: users, businesses, reviews are all json files
    """
    continue_filter = True
    filtered_users = set()
    filtered_businesses = set()
    while continue_filter:
        continue_filter = False
        # filter step 1
        users_interact_ind = {}
        business_interact_ind = {}
        for review in reviews:
            user_id = review['user_id'] # a list
            business_id = review['business_id'] # a list
            users_interact_ind[user_id] = users_interact_ind.get(user_id, 0) + 1
            business_interact_ind[business_id] = business_interact_ind.get(business_id, 0) + 1

        filtered_review_users = set(u for u in users_interact_ind.keys() if users_interact_ind[u]>=user_thresh)
        filtered_review_businesses = set(b for b in business_interact_ind.keys() if business_interact_ind[b]>=business_thresh)
        
        # loop until users' reviews equal to filtered reviews
        if (filtered_users != filtered_review_users) or (filtered_businesses != filtered_review_businesses):
            continue_filter = True

        # filter step 2
        # filter user and business
        # make user_friends_dict, only those users with lots of friends can be included
        user_friends_dict = {}
        for user in users:
            user_id = user['user_id']
            if user_id not in filtered_review_users:
                continue
            if not user['friends']:
                continue
            filtered_friends = [friend.strip() for friend in user['friends'].split(',') if friend.strip() in filtered_review_users]
            if len(filtered_friends) >= friend_thresh:
                user_friends_dict[user_id] = filtered_friends   # users with friends larger than friend_thresh

        continue_inside = True
        while continue_inside:
            friends = {}
            continue_inside = False
            for user, user_friends in user_friends_dict.items():
                filtered_friends = [friend for friend in user_friends if friend in user_friends_dict]   # friend in user_friends_dict's keys
                if len(filtered_friends) >= friend_thresh:
                    friends[user] = filtered_friends
                else:
                    continue_inside = True
            user_friends_dict = deepcopy(friends)   # this takes time

        filtered_users = set(user_friends_dict.keys())
        filtered_businesses_list = []

        for business in businesses:
            business_id = business['business_id']
            if business_id not in filtered_review_businesses:
                continue
            if not business['categories']:
                continue
            if not business['city']:
                continue
            filtered_businesses_list.append(business_id)
        filtered_businesses = set(filtered_businesses_list)

        filtered_review = []
        for review in reviews:
            if (review['user_id'] in filtered_users) and (review['business_id'] in filtered_businesses):
                filtered_review.append(review)
        reviews = deepcopy(filtered_review) # this takes time

        print(len(list(filtered_users)))
        print(len(list(filtered_businesses)))
        print(len(reviews))
        print('filter loop')

    print('filter complete')
    return filtered_users, filtered_businesses, filtered_review

def filter_rare_node_new(users, businesses, reviews, user_threshold, business_threshold, friend_threshold):
    continue_filter = True
    filtered_users = set()
    filtered_businesses = set()
    while(continue_filter):
        continue_filter = False
        # filter step 1
        users_posinteract_num = {}
        business_posinteract_num = {}
        users_neginteract_num = {}
        business_neginteract_num = {}
        for review in reviews:
            user_id = review['user_id']
            business_id = review['business_id']
            if review['stars'] > 3:
                users_posinteract_num[user_id] = users_posinteract_num.get(user_id, 0) + 1
                business_posinteract_num[business_id] = business_posinteract_num.get(business_id, 0) + 1
            else:
                users_neginteract_num[user_id] = users_neginteract_num.get(user_id, 0) + 1
                business_neginteract_num[business_id] = business_neginteract_num.get(business_id, 0) + 1
        user_interact = set(users_posinteract_num.keys()).intersection(set(users_neginteract_num.keys()))   # intersection
        business_interact = set(business_posinteract_num.keys()).intersection(set(business_neginteract_num.keys()))
        # filtered_review_users = set(u for u in user_interact if ((users_posinteract_num[u]+users_neginteract_num[u])>=user_threshold
        #                                                     and (users_posinteract_num[u]*users_posinteract_num[u])>0))
        # filtered_review_businesses = set(b for b in business_interact if ((business_posinteract_num[b]+business_neginteract_num[b])>=business_threshold
        #                                                     and (business_posinteract_num[b]*business_neginteract_num[b])>0))
        filtered_review_users = set(u for u in user_interact if (users_posinteract_num[u]>=user_threshold and users_neginteract_num[u])>=user_threshold)
        filtered_review_businesses = set(b for b in business_interact if (business_posinteract_num[b]>=business_threshold and business_neginteract_num[b])>=business_threshold)
        if (filtered_users != filtered_review_users) or (filtered_businesses != filtered_review_businesses):
            continue_filter = True
        # filter step 2
        #filter user and business
        user_friends_dict = {}
        for user in users:
            user_id = user['user_id']
            if user_id not in filtered_review_users:
                continue
            if not user['friends']:
                continue
            filtered_friends = [friend.strip() for friend in user['friends'].split(',') if friend.strip() in filtered_review_users]
            if len(filtered_friends) >= friend_threshold:
                user_friends_dict[user_id] = filtered_friends
        continue_inside = True
        while (continue_inside):
            friends = {}
            continue_inside = False
            for user, user_friends in user_friends_dict.items():
                filtered_friends = [friend for friend in user_friends if friend in user_friends_dict]
                if len(filtered_friends) >= friend_threshold:
                    friends[user] = filtered_friends
                else:
                    continue_inside = True
            user_friends_dict = deepcopy(friends)
        filtered_users = set(user_friends_dict.keys())
        filtered_businesses_list = []
        for business in businesses:
            business_id = business['business_id']
            if business_id not in filtered_review_businesses:
                continue
            if not business['categories']:
                continue
            if not business['city']:
                continue
            filtered_businesses_list.append(business_id)
        filtered_businesses = set(filtered_businesses_list)
        filtered_review = []
        for review in reviews:
            if (review['user_id'] in filtered_users) and (review['business_id'] in filtered_businesses):
                filtered_review.append(review)
        reviews = deepcopy(filtered_review)
        print(len(list(filtered_users)))
        print(len(list(filtered_businesses)))
        print(len(reviews))
        print('filter loop')
    print('filter complete')
    return filtered_users, filtered_businesses, filtered_review

if __name__ == '__main__':
    user_json     = load_jsondata_from_file('json/yelp_academic_dataset_user.json') # 25s   60.5s new
    business_json = load_jsondata_from_file('json/yelp_academic_dataset_business.json') # 4.45s   7.62s new
    review_json   = load_jsondata_from_file('json/yelp_academic_dataset_review.json')   # 69.8s    237.67s new 

    t0 = time.time()
    # filtered_user, filtered_business, filtered_reviews = filter_rare_node(user_json, business_json, review_json, 20, 20, 5)
    # filtered_user, filtered_business, filtered_reviews = filter_rare_node(user_json, business_json, review_json, 20, 20, 5)
    filtered_user, filtered_business, filtered_reviews = filter_rare_node_new(user_json, business_json, review_json, 7, 7, 3)
    filtered_user, filtered_business, filtered_reviews = filter_rare_node_new(user_json, business_json, review_json, 7, 7, 3)
    t1 = time.time()
    print("filter time cost:", t1 - t0) # 10min
    
    # save filtered results
    with open('filtered/users.pickle', 'wb') as fw:
        pickle.dump(filtered_user, fw, pickle.HIGHEST_PROTOCOL)
    with open('filtered/businesses.pickle', 'wb') as fw:
        pickle.dump(filtered_business, fw, pickle.HIGHEST_PROTOCOL)
    with open('filtered/reviews.pickle', 'wb') as fw:
        pickle.dump(filtered_reviews, fw, pickle.HIGHEST_PROTOCOL)