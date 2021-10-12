from common.make_ora_conn_pool import make_ora_pool
from common.make_pg_pool import make_pgbb_pool
from common.execute_query import execute_query

import pandas as pd
from tqdm import tqdm
import time
import pickle
import os
import csv


def load_data():

    if os.path.exists('data/graduate.pkl'):

        file = open('data/graduate.pkl', 'rb')
        course_access = pickle.load(file)
        file.close()
        return course_access

    start_dt = [
         '2021-03-01', '2021-04-01','2021-04-10','2021-04-20','2021-05-01','2021-06-01']

    end_dt = [
              '2021-04-01', '2021-04-10','2021-04-20','2021-05-1','2021-06-01','2021-06-18']

    pool_bb = make_pgbb_pool()
    conn_bb = pool_bb.getconn()

    start = time.time()
    bb_sql = open('../sql/pg/course_access.sql', 'r', encoding='utf-8').read()
    log_info = pd.DataFrame()
    for x in tqdm(range(1,6)):
        log = bb_sql.format(start_dt[x], end_dt[x])
        course_access = execute_query(conn_bb, log, 'N')
        course_access.to_pickle('data/course_access{}.pkl'.format(start_dt[x]))
        print(f'Time:{time.time() - start}')
    print(f'Time:{time.time()- start }')

    for x in tqdm(range(6)):
        file = open('data/course_access{}.pkl'.format(start_dt[x]), 'rb')
        course_access = pickle.load(file)
        file.close()
        log_info = log_info.append(course_access)
    log_info['date'] = pd.to_datetime(log_info['timestamp'], format='%Y-%m-%d')
    log_info['year_month'] = log_info['date'].dt.strftime('%Y-%m')


    pool_kuis = make_ora_pool()
    conn_kuis = pool_kuis.acquire()
    course_gpa = open('../sql/ora/course_gpa.sql', 'r', encoding = 'utf-8').read()
    std_course_gpa = execute_query(conn_kuis, course_gpa, 'N')

    log_count = log_info.groupby([ "event_type","city","company","department", "course_name","course_id","user_id","data"])["user_pk1"].agg('count').reset_index()
    df = log_count.merge(std_course_gpa, left_on='user_id', right_on = 'gpa021_std_id', how = 'left') #outer, campus 2 제거

    df.to_pickle('data/graduate.pkl')

