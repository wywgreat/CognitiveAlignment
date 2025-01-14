#!/usr/bin/env python
# -*- coding:utf-8 -*- #
import pandas as pd
import numpy as np

# # CSV文件列表
files = ['../origin/BBSR_gpt-3.5-turbo_OpenAI-1_score.csv',
         '../origin/BBSR_gpt-3.5-turbo_OpenAI-2_score.csv',
         '../origin/BBSR_gpt-3.5-turbo_OpenAI-3_score.csv',
         '../origin/BBSR_gpt-3.5-turbo_OpenAI-4_score.csv',
         '../origin/BBSR_gpt-3.5-turbo_OpenAI-5_score.csv'
         ]


# 读取Excel文件
data_list = []
for file in files:
    df = pd.read_csv(file)
    df['Temperature'] = df[['Hot', 'Cold']].max(axis=1)
    df = df.drop(columns=['Hot', 'Cold'])
    df['Texture'] = df[['Smooth', 'Rough']].max(axis=1)
    df = df.drop(columns=['Smooth', 'Rough'])
    df['Weight'] = df[['Light', 'Heavy']].max(axis=1)
    df = df.drop(columns=['Light', 'Heavy'])
    df.to_excel(file.split(".csv")[0]+"TTW.xlsx")
    data_list.append(df)

# 合并数据
combined_data = pd.concat(data_list, axis=0)
print(combined_data)
mean_df = combined_data.groupby("Concept").mean()

mean_df.to_excel('BBSR_gpt-3.5-turbo[MEAN]TTW.xlsx')

