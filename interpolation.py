# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 02:07:53 2023

@author: hal9000
"""
import pandas as pd
import numpy as np
df = pd.read_csv('input.csv')

    
df.reset_index(drop=True)

df1=df.head(4)
df_copy = pd.DataFrame().reindex_like(df1)
df_copy.reset_index(drop=True)
df_copy=df_copy.dropna()
lst15 =  [None] * 34
lst30 = [None] * 34
lst45 = [None] * 34
lst_previous = [None] * 34
for row in df1.itertuples():
    
    if(row.Index>0):
        
        # diff=pd.to_datetime( df1.loc[index].at["Date"])+ pd.Timedelta(minutes=15)
        # df.loc[len(df.index)] = ['Amy', 89, 93] 
        # print(diff)
        i=0;
        for (columnName, columnData) in df1.loc[row.Index].iteritems():
            if(i==0):
                
                lst15[0]=pd.to_datetime( columnData)+ pd.Timedelta(minutes=15)
                lst30[0]=pd.to_datetime( columnData)+ pd.Timedelta(minutes=30)
                lst45[0]=pd.to_datetime(columnData)+ pd.Timedelta(minutes=45)
                
            else:
                lst15[i]=lst_previous[i] + (columnData-lst_previous[i])/4
                lst30[i]=lst_previous[i] + (columnData-lst_previous[i])/2
                lst45[i]=lst_previous[i] + (columnData-lst_previous[i])*3/4
            lst_previous[i]=columnData
            
            i=i+1;
        print(lst15)
        # print(df_copy)
        # df_copy.append(pd.DataFrame([lst15],columns=list(df_copy)),ignore_index=True)
        df_copy = df_copy.append(pd.Series(lst_previous, index=df.columns[:len(lst_previous)]), ignore_index=True)
        df_copy = df_copy.append(pd.Series(lst15, index=df.columns[:len(lst15)]), ignore_index=True)
        df_copy = df_copy.append(pd.Series(lst30, index=df.columns[:len(lst30)]), ignore_index=True)
        df_copy = df_copy.append(pd.Series(lst45, index=df.columns[:len(lst45)]), ignore_index=True)
        # df_copy.append(pd.Series(lst15, index=df.columns[:len(lst15)]), ignore_index=True)
        # df_copy.loc[len(df_copy)]=[lst15]
        # print(lst15)
        # df_copy.loc[len(df.index)] = lst15
        # df_copy.loc[len(df.index)] =lst30
        # df_copy.loc[len(df.index)] = lst45    
        # print(row['Date'], row['VauDejes_temp'])
    else:
        i=0;
        for (_, columnData) in df1.loc[row.Index].iteritems():
            
            lst_previous[i]=columnData
            i=i+1;
        # for j in range(34):
            
        #     df_copy.loc[i,j] = lst_previous[j]
print(df_copy)
    
