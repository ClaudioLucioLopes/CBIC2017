# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:15:25 2017

@author: claud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



##################Leitura dos dados
#TODO: Colocar este método fora
df_serie = pd.read_csv('C:\\Users\\claud\\OneDrive\\CEFET\\2017-1 Com.Evol\\Trabalhos\\Artigo\\dados\\tab_9_22_train.csv',sep=';')
#nome das colunas
list(df_serie.columns.values)

#substitui valores nulos com 0
df_serie['chamadas'].fillna(0, inplace=True)

#df_serie =df_serie[0:100] 
#plot das serie
plt.plot(df_serie['chamadas'])
df_serie['chamadas'].describe()
#df_serie['chamadas'].hist()
lambda_cham = df_serie['chamadas'].mean()

df_serie['data'] = pd.to_datetime(df_serie['mes'].apply(str) +' '+df_serie['dia'].apply(str)+ ' ' +df_serie['Ano'].apply(str))


#########Cria Features para a serie em questao
df_serie['chamadas_mean'] = df_serie['chamadas'].mean()
df_serie['chamadas_std'] = df_serie['chamadas'].std()
df_serie['chamadas_min'] = df_serie['chamadas'].min()
df_serie['chamadas_max'] = df_serie['chamadas'].max()
df_serie['chamadas_q1'] = df_serie['chamadas'].quantile(0.25)
df_serie['chamadas_median'] = df_serie['chamadas'].median()
df_serie['chamadas_q3'] = df_serie['chamadas'].quantile(0.75)

#Identifica minimos e máximos locais  na serie
#Pode se o critério do desvio padrão ou talve uma não parametrica com a mediana
#Gera um variavel indicando se houve ou não pico na serie
from  peakdet import peakdetect
maxtab, mintab = peakdetect( (df_serie['chamadas']),df_serie['chamadas'].std())
ind = list(maxtab[:,0]) + list(mintab[:,0])
rep = [ 1 if w in ind else 0 for w in range(len( df_serie['chamadas'])) ]
df_serie['IND_max_min_peak'] = rep  





  
#TODO: Rever pois este pacote automatiza a parte de engineering featuring
#df_serie_hora = df_serie[['data','hora','chamadas']] 
#from tsfresh import extract_features,select_features
#from tsfresh.utilities.dataframe_functions import impute
#extracted_features = extract_features(df_serie_hora, column_id="hora", column_sort='data')
#impute(extracted_features)
#features_filtered = select_features(extracted_features, df_serie_hora.groupby('hora')['chamadas'].agg(np.mean))
#list(features_filtered.columns.values)

#Defasa a serie por Hora em um dia
hora_len = len(df_serie['hora'].unique())
for i in range(hora_len):
    df_serie['chamadas_lag'+str(i)] = df_serie['chamadas'].shift(i)

# media mediana e desvio da hora
df_hora = (df_serie.groupby('hora')['chamadas'].agg([np.median, np.mean,np.std])).reset_index()
df_hora.columns = ['hora','Chamadas_hora_median','Chamadas_hora_mean','Chamadas_hora_std']
df_serie = pd.merge(left=df_serie, right=df_hora, how='inner', left_on='hora',right_on='hora')
df_ind_hora=pd.get_dummies(df_serie['hora'],prefix='IND_hora')
df_serie = pd.concat([df_serie,df_ind_hora], axis=1, join_axes=[df_serie.index])
    
    
# media mediana e desvio do dia anterior e anterior ao anterior

df_dia_anterior = (df_serie.groupby('data')['chamadas'].agg([np.median, np.mean,np.std])).reset_index()
df_dia_anterior.columns = ['data','Chamadas_dia_median','Chamadas_dia_mean','Chamadas_dia_std']
df_serie = pd.merge(left=df_serie, right=df_dia_anterior, how='inner', left_on='data',right_on='data')

for i in range(2):#dia anterior e dia anterior ao anterior
    df_serie['Chamadas_dia_median_lag'+str(i)] = df_serie['Chamadas_dia_median'].shift(hora_len*(i+1))
    df_serie['Chamadas_dia_mean_lag'+str(i)] = df_serie['Chamadas_dia_mean'].shift(hora_len*(i+1))
    df_serie['Chamadas_dia_std_lag'+str(i)] = df_serie['Chamadas_dia_std'].shift(hora_len*(i+1))
 
# media mediana e desvio do dia da semana
df_DiaSemana = (df_serie.groupby('DiaSemana')['chamadas'].agg([np.median, np.mean,np.std])).reset_index()
df_DiaSemana.columns = ['DiaSemana','Chamadas_DiaSemana_median','Chamadas_DiaSemana_mean','Chamadas_DiaSemana_std']
df_serie = pd.merge(left=df_serie, right=df_DiaSemana, how='inner', left_on='DiaSemana',right_on='DiaSemana')
df_ind_dia_semana=pd.get_dummies(df_serie['DiaSemana'],prefix='IND_DiaSemana')
df_serie = pd.concat([df_serie,df_ind_dia_semana], axis=1, join_axes=[df_serie.index])



# media mediana e desvio do dia 
df_Dia = (df_serie.groupby('dia')['chamadas'].agg([np.median, np.mean,np.std])).reset_index()
df_Dia.columns = ['dia','Chamadas_dia_median','Chamadas_dia_mean','Chamadas_dia_std']
df_serie = pd.merge(left=df_serie, right=df_Dia, how='inner', left_on='dia',right_on='dia')
df_ind_dia=pd.get_dummies(df_serie['dia'],prefix='IND_dia')
df_serie = pd.concat([df_serie,df_ind_dia], axis=1, join_axes=[df_serie.index])
    



df_serie.sort_index(by=['mes','dia'], ascending=[ False,False])
df_serie.to_csv('c:\\temp\\featured_base.csv')
