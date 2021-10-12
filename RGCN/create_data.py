# 학생 - 데이터 클릭
#
# 학생 전공
# 학생 이중전공?
# 학생 수업 - 수강
# 학생 수업  사전수강
# 학생 수업 전공, 교양
# 수업 전공 과목학과
import csv
import pandas as pd

def load_from_file():
    # with open('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn수강.csv', encoding='euc-kr',
    #           mode='r') as f:
    #     entities_dict = dict()
    #     data = csv.reader(f, delimiter=' ', quotechar='|')
    #     for row in data:
    #         print(', '.join(row))

    #entities 학생 전공 수업
    #relations 수강, 사전수강, 과목특성, 전공, 이중전공?

    entities_dict = dict()

    data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn수강.csv', encoding='euc-kr')
    data = data[['COR011_STD_ID', '학생학과', '과목학과','과목명']]
    data = pd.concat([data['COR011_STD_ID'].squeeze(),data['학생학과'].squeeze(), data['과목학과'].squeeze(),data['과목명'].squeeze()]).drop_duplicates()

    pre_data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn사전수강.csv', encoding='euc-kr')
    pre_data = pre_data[['COR015_STD_ID', '학생학과', '과목학과','과목명_1']]
    pre_data = pd.concat([pre_data['COR015_STD_ID'].squeeze(),pre_data['학생학과'].squeeze(), pre_data['과목학과'].squeeze(),pre_data['과목명_1'].squeeze()]).drop_duplicates()

    entities = pd.concat([data,pre_data]).drop_duplicates()

    for i, entity in enumerate(entities):
        entities_dict[i] = entity

    with open("/data/workspace/holly0015/test_project1/project2/RGCN/경영data/entities.dict", 'w') as f:
        for key, value in entities_dict.items():
            f.write('%s\t%s\n' % (key, value))

    relations_dict = dict()
    relations_dict[0] = '수강'
    relations_dict[1] = '사전수강'
    relations_dict[2] = '전공'

    data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn수강.csv', encoding='euc-kr')
    data = data['과목특성'].drop_duplicates()

    pre_data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn사전수강.csv', encoding='euc-kr')
    pre_data = pre_data['과목특성'].drop_duplicates()
    relations = pd.concat([data, pre_data]).drop_duplicates()

    for i, relation in enumerate(relations):
        relations_dict[i+3] = relation

    with open("/data/workspace/holly0015/test_project1/project2/RGCN/경영data/relations.dict", 'w') as f:
        for key, value in relations_dict.items():
            f.write('%s\t%s\n' % (key, value))

    #train, valid, test
    data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn수강.csv', encoding='euc-kr')
    pre_data = pd.read_csv('/data/workspace/holly0015/test_project1/project2/data/경영Rgcn사전수강.csv', encoding='euc-kr')
    data = data[['COR011_STD_ID', '과목명','과목특성']]
    pre_data = pre_data[['COR015_STD_ID', '과목명_1','과목특성']]
    pre_data = pre_data.rename(columns={'COR015_STD_ID':'COR011_STD_ID','과목명_1':'과목명'})
    data = pd.concat([data,pre_data]).drop_duplicates().reset_index()

    with open("/data/workspace/holly0015/test_project1/project2/RGCN/경영data/train.txt", 'w') as f:
        for i in range(25000):
            f.write('%s\t%s\t%s\n' % (data['COR011_STD_ID'][i], data['과목특성'][i],data['과목명'][i]))

    with open("/data/workspace/holly0015/test_project1/project2/RGCN/경영data/valid.txt", 'w') as f:
        for i in range(25000,29000):
            f.write('%s\t%s\t%s\n' % (data['COR011_STD_ID'][i], data['과목특성'][i],data['과목명'][i]))

    with open("/data/workspace/holly0015/test_project1/project2/RGCN/경영data/test.txt", 'w') as f:
        for i in range(29000, len(data)):
            f.write('%s\t%s\t%s\n' % (data['COR011_STD_ID'][i], data['과목특성'][i],data['과목명'][i]))

    # for i in range(len(data)):
    #     if data["COR011_STD_ID"][i] in entities_dict:
    #         entities_dict[data["COR011_STD_ID"][i]].add(data["학생학과"][i])
    #     else:
    #         set_ = set()
    #         set_.add(data['학생학과'][i])
    #         entities_dict[data['COR011_STD_ID'][i]] = set_

    # for e in entities_dict.keys():
    #     entities_dict[e] = list(entities_dict[e])
    #
    # print(entities_dict.values())
    return