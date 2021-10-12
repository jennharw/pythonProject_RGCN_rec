from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns

import random
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from surprise import Reader, Dataset
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV

from utils import load_data

random.seed(1234)
np.random.seed(1234)

df = load_data()
df = df.dropna()


def eda(df):
    print(df.head())
    print(df.describe()['AVG_SCORE'])

    # EDA
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x='AVG_SCORE', data=df)

    ax.set_yticklabels([num for num in ax.get_yticks()])

    plt.tick_params(labelsize=15)
    plt.title("Distribution of Ratings in train data", fontsize=20)
    plt.xlabel("Ratings", fontsize=20)
    plt.ylabel("Number of Ratings(Millions)", fontsize=20)
    plt.show()

    no_of_rated_lectures_per_user = df.groupby(by="LEC411_STD_ID")['AVG_SCORE'].count().sort_values(ascending=True)
    print(no_of_rated_lectures_per_user.head())
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    sns.kdeplot(no_of_rated_lectures_per_user.values, shade=True, ax=axes[0])
    axes[0].set_title("PDF", fontsize=18)
    axes[0].set_xlabel("Number of Ratings by user", fontsize=18)
    axes[0].tick_params(labelsize=15)
    sns.kdeplot(no_of_rated_lectures_per_user.values, shade=True, cumulative=True, ax=axes[1])
    axes[1].set_title("CDF", fontsize=18)
    axes[1].set_xlabel("Number of Ratings by user", fontsize=18)
    axes[1].tick_params(labelsize=15)
    fig.subplots_adjust(wspace=2)
    plt.tight_layout()
    plt.show()

    no_of_ratings_per_lectures = df.groupby(by='LEC411_COUR_CD')['AVG_SCORE'].count().sort_values(ascending=True)

    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_axes([0.1, 0.1, 1, 1])
    plt.title("Number of Ratings Per Lectrue", fontsize=20)
    plt.xlabel("Lecture", fontsize=20)
    plt.ylabel("Count of Ratings", fontsize=20)
    plt.plot(no_of_ratings_per_lectures.values)
    plt.tick_params(labelsize=15)
    axes.set_xticklabels([])
    plt.show()

    return


def user_item_matrix(df):
    lec_matrix = pd.pivot_table(df, index='LEC411_STD_ID', columns='LEC411_COUR_CD', values='AVG_SCORE',
                                fill_value=0).reset_index()
    print(lec_matrix.shape)
    rows, cols = lec_matrix.shape
    presentElements = (lec_matrix != 0).sum(1).sum()
    print("Sparsity Of Train matrix : {}% ".format((1 - (presentElements / (rows * cols))) * 100))
    print("Global Average Rating {}".format(lec_matrix.sum(1).sum() / presentElements))

    sumOfRatings = lec_matrix.sum(1)
    noOfRatings = (lec_matrix != 0).sum(1)
    rows, cols = lec_matrix.shape
    AvgRatingUser = {i: sumOfRatings[i] / noOfRatings[i] for i in range(rows) if noOfRatings[i] != 0}
    print(f"Average rating of user 25 = {AvgRatingUser[25]}")

    sumOfRatings = lec_matrix.sum(0)
    noOfRatings = (lec_matrix != 0).sum(0)
    rows, cols = lec_matrix.shape
    AvgRatingLecture = {i: sumOfRatings[i] / noOfRatings[i] for i in range(1, cols) if noOfRatings[i] != 0}
    print(f"Average rating of user 25 = {AvgRatingLecture[25]}")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    fig.suptitle('Avg Ratings per User and per Lecture', fontsize=25)

    user_average = [rats for rats in AvgRatingUser.values()]
    sns.distplot(user_average, hist=False, ax=axes[0], label="PDF")
    sns.kdeplot(user_average, cumulative=True, ax=axes[0], label="CDF")
    axes[0].set_title("Average Rating Per User", fontsize=20)
    axes[0].tick_params(labelsize=15)
    axes[0].legend(loc='upper left', fontsize=17)

    movie_average = [ratm for ratm in AvgRatingLecture.values()]
    sns.distplot(movie_average, hist=False, ax=axes[1], label="PDF")
    sns.kdeplot(movie_average, cumulative=True, ax=axes[1], label="CDF")
    axes[1].set_title("Average Rating Per Movie", fontsize=20)
    axes[1].tick_params(labelsize=15)
    axes[1].legend(loc='upper left', fontsize=17)

    plt.subplots_adjust(wspace=0.2, top=0.85)
    plt.show()

    # cold start problem


def compute_item_similarity(df):
    print(df.head(5))
    lec_matrix = pd.pivot_table(df, index='LEC411_COUR_CD', columns='LEC411_STD_ID', values='AVG_SCORE',
                                fill_value=0)  # .reset_index()
    cos = cosine_similarity(lec_matrix, lec_matrix)
    top50_indices = cos.argsort()[-50:]
    top50_similar = cos[top50_indices]
    lec_matrix2 = pd.pivot_table(df, index='LEC411_COUR_CD', columns='LEC411_STD_ID', values='AVG_SCORE',
                                 fill_value=0).reset_index()

    indices = pd.Series(lec_matrix2.index, index=lec_matrix.index)

    idx = indices['BUSS475']
    scores = list(enumerate(cos[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    similar_30 = scores[:30]

    plt.figure(figsize=(10, 8))
    plt.plot([i[1] for i in scores], label="All Similar")
    plt.plot([i[1] for i in similar_30], label="Top 30 Similar Movies")
    plt.title("Similar Movies to Marketing Of Internet", fontsize=25)
    plt.ylabel("Cosine Similarity Values", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)
    plt.show()

    scores = scores[1:50]
    major_indices = [i[0] for i in scores]

    result = indices[major_indices].to_frame()  # .index#['LEC411_STD_ID']
    info_std = df[['LEC411_COUR_CD', '수업명']].drop_duplicates()
    rec = pd.merge(result, info_std, on='LEC411_COUR_CD', how='inner')

    return rec


def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])
    return actual, predicted


def get_error(predictions):
    actual, predicted = get_ratings(predictions)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(abs((actual - predicted) / actual)) * 100
    return rmse, mape


def run_surprise(algo, trainset, testset):
    train = dict()
    test = dict()

    algo.fit(trainset)

    train_pred = algo.test(trainset.build_testset())

    train_actual, train_predicted = get_ratings(train_pred)
    train_rmse, train_mape = get_error(train_pred)
    print("RMSE = {}".format(train_rmse))
    print("MAPE = {}".format(train_mape))
    train = {"RMSE": train_rmse, "MAPE": train_mape, "Prediction": train_predicted}

    print("TEST DATA")
    test_pred = algo.test(testset)
    test_actual, test_predicted = get_ratings(test_pred)
    test_rmse, test_mape = get_error(test_pred)
    print("RMSE = {}".format(test_rmse))
    print("MAPE = {}".format(test_mape))
    print("-" * 50)
    test = {"RMSE": test_rmse, "MAPE": test_mape, "Prediction": test_predicted}

    return train, test


def svd_train():
    df = pd.read_csv('../../data/ClubRating.csv')
    print(len(df))

    train_set = df[['userid', 'club', 'ratings']][:530]
    test_set = df[['userid', 'club', 'ratings']][520:]

    reader = Reader(rating_scale=(0, 6))
    data = Dataset.load_from_df(train_set, reader)
    trainset = data.build_full_trainset()
    testset = list(
        zip(test_set["userid"].values, test_set["club"].values, test_set["ratings"].values))

    param_grid = {'n_factors': [5, 7, 10, 15, 20, 25, 35, 50, 70, 90]}
    param_grid = {'n_factors': [10, 30, 50, 80, 100], 'lr_all': [0.002, 0.006, 0.018, 0.054, 0.10]}
    #gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
    #gs.fit(data)
    # best RMSE score
    #print(gs.best_score['rmse'])
    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])

    #algo = SVD(n_factors=gs.best_params['rmse']['n_factors'], biased=True, verbose=True)  # NMF, SVDpp
    #algo = SVDpp(n_factors=gs.best_params['rmse']['n_factors'], lr_all=gs.best_params['rmse']["lr_all"], verbose=True)
    #train_result, test_result = run_surprise(algo, trainset, testset)
    #print("SVDpp {}".format(train_result, test_result))

    # predictions = algo.test(testset)
    # actual = np.array([pred.r_ui for pred in predictions])
    # predicted = np.array([pred.est for pred in predictions])

    algo = SVDpp(n_factors=100, lr_all=0.002, verbose=True)
    algo.fit(trainset)
    print(algo.predict('abgc4531','로타랙트'))
    print(algo.predict('bionda7','로타랙트'))

    #print(recom_lec(algo, 2020120092, top_n=15, unseen=True))

# 2019120449 2018130430  2020120092
def recom_lec(algo, userid, top_n=False, unseen=True):
    # 전체 item id
    info_std = df[['LEC411_COUR_CD', '수업명']].drop_duplicates()

    total_items = info_std['LEC411_COUR_CD'].tolist()
    seen_items = df[df['LEC411_STD_ID'] == userid]['LEC411_COUR_CD'].tolist()
    unseen_items = [items for items in total_items if items not in seen_items]

    # 평점을 내리지 않은 item에 대해 평점 예측하는 경우
    if unseen:
        predictions = [algo.predict(userid, itemId) for itemId in unseen_items]
    # 평점을 이미 내린 item에 대해 평점 예측하는 경우
    else:
        predictions = [algo.predict(userid, itemId) for itemId in seen_items]

    # 예측평점(est) 기준으로 정렬하는 함수
    predictions.sort(key= lambda pred: pred.est, reverse=True)
    # 상위 n개 결과만을 도출하는 경우
    if top_n:
        predictions = predictions[:top_n]

    top_item_ids = [pred.iid for pred in predictions]
    cnt = 0
    for i in top_item_ids:
        name = info_std[info_std['LEC411_COUR_CD'] == i]['수업명']
        if cnt == 0:
            top_item_titles = name
            cnt += 1
        else:
            top_item_titles = pd.concat([top_item_titles, name])
    return top_item_titles
