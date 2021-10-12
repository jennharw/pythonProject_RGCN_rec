import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import io

from data_load_from_db import load_data

def make_gexf(course = None): #i 특정 과목, 아니면 x
    df = load_data()
    df = df.rename(columns={'user_pk1': 'count'})
    if course != None:
      data = df[(df['gpa021_cour_cd'] == course) & (df['count'] > 2 )]
    else:
      data = df[df['count'] >1]

    #node ; source target weight

    df_cs = data[['user_id', 'data', 'count']]
    df_cs = df_cs.rename(columns={'user_id':'Source', 'data':'Target'})
    dataset1 = df_cs.groupby(['Source', 'Target'])['count'].sum().reset_index().rename(columns = {'count':'Weight'})

    node1 = dataset1[['Source', 'Source']].drop_duplicates().rename(columns={'Source': 'Fullname', 'Source': 'Id'})
    node2 = dataset1[['Target', 'Target']].drop_duplicates().rename(columns={'Target': 'Fullname', 'Target': 'Id'})
    nodelist = pd.concat([node1, node2], axis=0).drop_duplicates().set_index('Id').T.to_dict()

    #network generation
    G_imp = nx.from_pandas_edgelist(dataset1, 'Source', 'Target', create_using = nx.DiGraph(), edge_attr = 'Weight')
    nx.set_node_attributes(G_imp, nodelist) #nodelist 추가
    nx.write_gexf(G_imp, r"data/course_access_%s.gexf" % course)

    #node attribute
    node_attr = data[['user_id', 'gpa021_grade', 'gpa021_cour_nm', 'company']].drop_duplicates()
    node_attr = node_attr.rename(columns={'user_id':'Id', 'gpa021_grade':'Attribute'})
    node_attr = node_attr.set_index('Id')
    #click label
    node_attr2 = data[['data', 'data']].drop_duplicates()
    node_attr2.columns = ['Id', 'Label']

    node_attr2 = node_attr2.set_index('Id')
    gr_attr = pd.concat([node_attr, node_attr2], axis=0)
    gr_attr.to_excel('data/node_attr_%s .xlsx' % course)


    return "완료"


def count_click():
    df = load_data()
    df = df.rename(columns={'user_pk1': 'count'})

    #df.groupby(['gpa021_grade'])['count'].sum().plot(kind='bar')

    mdf =  df.groupby(['gpa021_grade','data'])['count'].mean().sort_values(ascending=False).head(1000).reset_index()
    mdf.to_excel('data/dataByGroup.xlsx')

    import matplotlib.font_manager as fm

    font_location = '/data/workspace/holly0015/test_project1/project2/src/NanumGothic.ttf'
    font_name = fm.FontProperties(fname = font_location).get_name()
    plt.rc('font', family=font_name)
    print(f"설정 폰트 글꼴: {plt.rcParams['font.family']}, 설정 폰트 사이즈: {plt.rcParams['font.size']}")

    return