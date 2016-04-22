# -*- coding:utf8 -*-
from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

movies = datasets.load_sample_movies()
songs = datasets.load_sample_songs()

##print 'movies user_id:',movies.user_ids
##print 'movies user data:',movies.data
##print 'moves item_ids:',movies.item_ids
"""
print '==========movies data============'
print 'user_ids :', movies.user_ids
print 'user_data:', movies.data
print 'item_ids :', movies.item_ids
print '==========movies data============'
"""
"""
#data format of movies
user_ids : {1: 'Jack Matthews', 2: 'Mick LaSalle', 3: 'Claudia Puig', 4: 'Lisa Rose', 5: 'Toby', 6: 'Gene Seymour', 7: 'Michael Phillips'}
user_data: {
1: {1: 3.0, 2: 4.0, 3: 3.5, 4: 5.0, 5: 3.0        },
2: {1: 3.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 2.0},
3: {        2: 3.5, 3: 2.5, 4: 4.0, 5: 4.5, 6: 3.0},
4: {1: 2.5, 2: 3.5, 3: 2.5, 4: 3.5, 5: 3.0, 6: 3.0},
5: {        2: 4.5, 3: 1.0, 4: 4.0                },
6: {1: 3.0, 2: 3.5, 3: 3.5, 4: 5.0, 5: 3.0, 6: 1.5},
7: {1: 2.5, 2: 3.0,         4: 3.5, 5: 4.0        }}
item_ids : {1: 'Lady in the Water', 2: 'Snakes on a Planet', 3: 'You, Me and Dupree', 4: 'Superman Returns', 5: 'The Night Listener', 6: 'Just My Luck'}
"""

model = MatrixPreferenceDataModel(movies.data)   #build the model
similarity = UserSimilarity(model, pearson_correlation)  #build the similarity
recommender = UserBasedRecommender(model, similarity, with_preference=True)   #build the recommender

# print recommender.recommend(3)
# print recommender.recommend(1, how_many=None, minimal_similarity=5.0)
print recommender.recommend(1, how_many=100, model__minimal_similarity=0.5)
# print recommender.most_similar_users(1)




