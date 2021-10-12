#from utils import load_data
#from modelSVD import eda, user_item_matrix, compute_item_similarity, svd_train, recom_lec

from eda_gephi import make_gexf, count_click
def print_hi(name):

    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    #make_gexf("BUSS342")
    #count_click()
    
    #RGCN
    print("---------------RGCN Log 성적예측, 클릭추천------------------")
    print("---------------RGCN +glove 과목 추천 ------------------")
    print("---------------svd------------------")


    #df = df.dropna()
    #eda(df)
    #user_item_matrix(df)
    #similarity_compute(df)
    #svd_train()
    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
