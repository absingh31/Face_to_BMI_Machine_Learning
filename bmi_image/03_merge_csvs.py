import pandas as pd

df_hw = pd.read_csv('height_weight.csv')
df_img = pd.read_csv('img_scores.csv')

df_hw['file_name'] = df_hw['file_name'].str.replace('.json','')
df_img['image_name'] = df_img['image_name'].str.replace('.jpeg','')

df_all=df_img.merge(df_hw,how='left',left_on='image_name',right_on='file_name')
df_all.to_csv('bmi_vs_score.csv')