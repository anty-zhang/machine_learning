from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender
import time

def testNan():
    import numpy as np
    pre = [(5838777334761013530, np.nan), (5855107930144739859, np.nan)]

    for (item, pre) in pre:
        # print 'item=', item, ',pre=', pre
        if np.isnan(pre):
            print 'item=', item

def testParam(how_many=None, **param):
    print 'how_many:', how_many
    print 'param:', param

if __name__ == '__main__':
    # testParam(how_many=12, tt=10, dd=40)
    start_time = time.time()
    file_name = 'score_data.txt'
    with open(file_name) as f:
        d = f.read()
    # print 'd=', d
    d = eval(d)
    model = MatrixPreferenceDataModel(d)   #build the model
    similarity = UserSimilarity(model, pearson_correlation, num_best=50)  #build the similarity
    recommender = UserBasedRecommender(model, similarity, with_preference=True)   #build the recommender

    print recommender.recommend(user_id=31071, how_many=30, minimal_similarity=0.8)
    # print recommender.recommend(20832)
    # print recommender.most_similar_users(user_id=31071)
    # print d[20832]
    print '====recommand time:', time.time() - start_time
    '''
    for key in d:
        print '===key===', key
        print recommender.recommend(key)
    '''