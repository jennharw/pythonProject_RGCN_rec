import pandas as pd
import numpy as np
import argparse
from scipy.stats.mstats import gmean, hmean
import itertools
import re
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

from datetime import datetime
from tqdm import tqdm
import ast

import psycopg2 as pg
import cx_Oracle

import pickle
import gzip



def data_load(filename, cursor):
    product = cursor
    f = open(filename, 'r')
    
    text = ''
    while True:
        line = f.readline()        
        if not line: break
        a = str(line)
        text = text + a
    f.close()
    
    data = pd.read_sql(text, product) 
    print("#------Read SQL Completed!------#")
    return data


def connect():
    user = 'datahub'
    password = 'datahub123!@#'
    host_product = '163.152.11.12'
    dbname = 'pkuhub'
    port = '5432'

    product_connection_string = "dbname={dbname} user={user} host={host} password={password} port={port}"\
                                .format(dbname=dbname,
                                        user=user,
                                        host=host_product,
                                        password=password,
                                        port=port)
    try:
        product = pg.connect(product_connection_string)
    except:
        print('*****ERROR******')

        pc = product.cursor()
    return product
    

def prep():
    """
    since input data was intersectino of 'wish' and 'register' data, 
    courses that were registered but not wished should be eliminated in the recommendation list
    """
    product = connect()
    filter_reg = data_load("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Course Recom/RGCN/sql/course_reg.txt", product) #수강한 과목
    rgcn = data_load("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Course Recom/RGCN/sql/rgcn_elec_now_open.txt", product) #rgcn
    
    product.close()
    
    filter_reg = filter_reg[['std_id','cour_cd']].drop_duplicates()
    filter_reg['key'] = filter_reg['std_id'] + filter_reg['cour_cd']
    drp_list = filter_reg['key'].tolist()
    del filter_reg

    rgcn['key'] = rgcn['std_id']+rgcn['cour_cd']
    rgcn_f = rgcn[~rgcn['key'].isin(drp_list)][['std_id','cour_div_nm','credit','cour_cd','cour_nm','score']]
    del rgcn
           
    return rgcn_f

class Recommend:

    def __init__(self):
        self.rgcn_f = prep()
        
    #최초페이지용 함수
    def initial_load(self, std_id):
        first_rec = self.rgcn_f[self.rgcn_f['std_id']==std_id][['std_id','cour_div_nm','credit','cour_cd','cour_nm','score']]
        return first_rec.sort_values(by ='score', ascending =False).iloc[:21] 

    #추천받기 누른 후 함수
    def final_score(self, std_id, click_list):
        product = connect()
        first_rec = self.initial_load(std_id)
        
        q = f"""
            SELECT
            DISTINCT DAT.CLICK_COUR_CD
            , DAT.COUR_CD
            , DAT.SCORE
            , KLT.COUR_DIV_NM
            , KLT.COUR_NM COUR_NM
            , KLT.CREDIT CREDIT
        FROM
            RECSYS.DH_ADV911TL DAT, RECSYS.DH_ADV100TL KLT
        WHERE
            1 = 1
            AND DAT.COUR_CD = KLT.COUR_CD
            AND KLT.COUR_DIV = '01'
            AND KLT.YR BETWEEN '2017' AND '2021'
            AND KLT.REC_USE_YN = 'Y'
            AND CLICK_COUR_CD IN (
                SELECT
                        COR011_COUR_CD
                FROM
                        SRC.KUPID_COR011TL
                WHERE
                        COR011_STD_ID = '{std_id}')
                  """
        
        cour_score = pd.read_sql(q, product) #(course, co_taken, course) source nodes are from a student's course history
        reg_list = cour_score.click_cour_cd.unique().tolist()      
        product.close()
        
        print("#------", std_id,"'s choice is...", click_list,"------#")
        
        #similarity filtering clicked courses 
        cour_score = cour_score[(cour_score['click_cour_cd'].isin(click_list))][['cour_cd','cour_nm','cour_div_nm','credit','score']]
        cour_score = cour_score[~cour_score['cour_cd'].isin(reg_list)]
        cour_score['score'] = cour_score['score'].astype(float)*5

        final_rec = pd.concat([first_rec[['cour_cd','cour_nm','cour_div_nm','credit','score']], cour_score], axis=0)
        
        
        #final_rec = final_rec.groupby(['cour_cd','cour_div_nm','cour_nm','credit']).apply(gmean).reset_index().rename(columns={0:'score'})
        #final_rec = final_rec.groupby(['cour_cd','cour_div_nm','cour_nm','credit']).apply(hmean).reset_index().rename(columns={0:'score'})
        final_rec = final_rec.groupby(['cour_cd','cour_div_nm','cour_nm','credit']).mean().reset_index().rename(columns={0:'score'})
        
     
        final_rec.score = final_rec.score.apply(lambda x: float(x))
        final_rec = final_rec.sort_values(by='score', ascending=False)
        final_rec['std_id'] = std_id
       
        return final_rec.sort_values(by ='score', ascending =False)[['std_id','cour_div_nm','credit','cour_cd','cour_nm','score']].iloc[:21]       
    
    def course_rec(self, std_id, click_list):
        if len(click_list) == 0:
            rec_list = self.initial_load(std_id)
        else:
            rec_list = self.final_score(std_id, click_list)
        return print(rec_list)