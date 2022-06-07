
import pandas as pd
import numpy as np


# In[para]
max_count = 300
lower_prob = 0.80
top = 1

# In[查看平衡狀況]
df_pseudo = pd.read_csv('csv/student_img_pseudo_label_step1.csv')
#df_pseudo = df_pseudo[13000:]

for i in range(219):
    Total = df_pseudo[str(i)].sum()
    print (f"Column {i} sum:",Total)

# In[]
label_cols = df_pseudo.columns[1:].to_list()
y_pred = np.argmax(np.array(df_pseudo[label_cols]), axis=1)

# Top n sum
a = np.array(df_pseudo[label_cols])
arr1 = np.argsort(-a, kind='mergesort', axis=1).argsort() < top
max_prod = a[arr1]
max_prod = np.reshape(max_prod, (len(df_pseudo), top))
max_prod = max_prod.sum(axis=1)
# Top 1 
# max_prod = np.max(df_pseudo, axis=1)

df_pseudo['y_pred'] = y_pred
df_pseudo['max_prod'] = max_prod

# In[篩選概率大於lower_prob]
print('總數:', len(df_pseudo),'篩選後:',sum(max_prod>=lower_prob))

df_pseudo = df_pseudo[max_prod>=lower_prob]

# In[]
# 再次驗證看看
for i in range(219):
    Total = df_pseudo[str(i)].sum()
    print (f"Column {i} sum:",Total)

# In[篩選max_count]
col_sum = df_pseudo.groupby('y_pred')['filepath'].count()

df_pseudo['rank_sel'] = 0
for i in range(219):
    i = str(i)
    rank = df_pseudo[i].rank(ascending=False)
    df_pseudo['rank_sel'] += (rank<=max_count).astype(int)
    
df_pseudo = df_pseudo[df_pseudo['rank_sel'] > 0]
del df_pseudo['rank_sel']

# In[]
df_pseudo.to_csv('csv/student_img_pseudo_label_step1_filter.csv')
