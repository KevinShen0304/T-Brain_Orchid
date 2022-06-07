'''
概念:預計各種都有X張以上的照片
作法:
    從總概率最小的類別開始，排序選前X個做為該類
    直到遍歷所有種類，剩下的再依照原本的分類區分
'''

import pandas as pd
import numpy as np

# In[para]
lower_count = 90 #90
lower_prob = 0.10
lower_gap = -0.10
label_list = [str(i) for i in range(219)]

# In[]
#df = pd.read_csv('csv/predict/noisy_student_predict_prob.csv')
df = pd.read_csv('csv/predict/predict_prob.csv')

label_sum_prod = {"label":[],"sum_prod":[]}
for i in label_list:
    Total = df[i].sum()
    label_sum_prod["label"].append(i)
    label_sum_prod["sum_prod"].append(Total)

label_sum_prod = pd.DataFrame(label_sum_prod)

# In[挑選]
label_sum_prod['rank'] = label_sum_prod['sum_prod'].rank(ascending=True)

# 從sum_prod最小的開始
label_sum_prod = label_sum_prod.sort_values(by=['rank']).reset_index(drop=True)

result1 = pd.DataFrame()
for i, row in label_sum_prod.iterrows():
    label = row['label']

    df = df.sort_values(by=[label], ascending=False)
    filename_list = []
    labels_list = []
    for j, r in df.iterrows():
        filename = r['filename']
        del r['filename']

        # 至少大於某概率
        if r[label] < lower_prob:
            # 沒達到lower_count
            result_sel = pd.DataFrame({'filename':filename_list,'category':labels_list})
            result1 = result1.append(result_sel)
            # 選完後從df篩除
            df_sel = result_sel['filename']
            df = df[~df['filename'].isin(df_sel)]
            
            min_prob = r[label] 
            print('label:', label, 'count:', len(result_sel),'min_prob:', min_prob)

            break # 跳出
        
        # 不能與最大概率差太多
        max_prod = max(list(r))
        gap = r[label] - max_prod
        if gap < lower_gap:
            continue

        if gap < 0:
            print(filename,'change_prob:',r[label],'max_prod:', max_prod)
        
        filename_list.append(filename)
        labels_list.append(label) 
        
        # 達到lower_count
        if len(filename_list)== lower_count:
            result_sel = pd.DataFrame({'filename':filename_list,'category':labels_list})
            result1 = result1.append(result_sel)
            # 選完後從df篩除
            df_sel = result_sel['filename']
            df = df[~df['filename'].isin(df_sel)]
            
            min_prob = r[label] 
            print('label:', label, 'count:', len(result_sel),'min_prob:', min_prob)

            break
    
            
# In[補上剩下的]
print('剩餘數量:',len(df))
y_pred = np.argmax(np.array(df[label_list]), axis=1)
result2 = df[['filename']]
result2['category'] = y_pred

# In[組合]
result = pd.concat([result1,result2], axis = 0)
result = result.sort_index()

result.to_csv('csv/predict_label_banlance_90_1010.csv', index=False)

# =============================================================================
# # In[對照兩者差異數量]
# predict_label_banlance = pd.read_csv('csv/predict_label_banlance_1010.csv')
# predict_label = pd.read_csv('csv/predict_label.csv')
# 
# comp = predict_label_banlance.merge(predict_label, on='filename')
# comp = comp[comp['category_x']!=comp['category_y']]
# # category_x=修改後，category_y=category_y=修改前
# print(comp)
# 
# df = pd.read_csv('csv/predict_prob.csv')
# change = df[df['filename']=='gbr8o6p2zs.jpg'][['filename','55','121']]
# print(change)
# 
# =============================================================================
# In[]
# =============================================================================
# result1 = pd.DataFrame()
# for i, row in label_sum_prod.iterrows():
#     label = row['label']
# 
#     result_sel = df.nlargest(n=lower_count, columns=[label])
# 
#     # 至少大於某概率
#     result_sel = result_sel[result_sel[label]>lower_prob]
#     # 不能與最大概率差太多
#     max_prod = np.max(result_sel, axis=1)
#     gap = result_sel[label] - max_prod
#     result_sel = result_sel[gap >= lower_gap]
#     
#     min_prob = min(result_sel[label])
#     print('label:', label, 'count:', len(result_sel),'min_prob:', min_prob)
#     for i in gap.index:
#         if gap[i] < 0:
#             print(result_sel['filename'][i],'change_prob:',result_sel[label][i],'max_prod:', max_prod[i])
#     
#     result_sel = result_sel[['filename']]
#     result_sel['category'] = label
#     result1 = result1.append(result_sel) 
#     # 選完後從df篩除
#     df_sel = result_sel['filename']
#     df = df[~df['filename'].isin(df_sel)]
# =============================================================================
