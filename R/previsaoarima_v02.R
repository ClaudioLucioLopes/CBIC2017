library(dummies)
library(forecast)

#############################################
#Leitura dos arquivos e tratamento para valores nulos
#############################################

df_train<-read.csv2('C:\\temp\\tab_9_22_train.csv',sep=';')
df_teste<-read.csv2('C:\\temp\\tab_9_22_test.csv',sep=';')
df_train[is.na(df_train)] <- 0
df_teste[is.na(df_teste)] <- 0

df_teste$hora<-as.factor(df_teste$hora)
df_train$hora<-as.factor(df_train$hora)
df_teste$dia<-as.factor(df_teste$dia)
df_train$dia<-as.factor(df_train$dia)

#############################################
#Criação das Features de hora e dia de acordo com a matriz binária
#############################################

df_teste_dummy <- dummy.data.frame(df_teste, sep = "_")
df_train_dummy <- dummy.data.frame(df_train, sep = "_")
names(df_train_dummy)
names(df_teste_dummy)

dummy_df <- df_train_dummy[,setdiff(names(df_train_dummy),c('dia_1','hora_13','mes','Ano','DiaSemana','chamadas'))]

#############################################
#Processo de estimação dos coeficiente via máxima verossimilhança
#############################################

model<-auto.arima(y = df_train_dummy$chamadas,xreg = dummy_df)
summary(model)
save(model,file='C:\\Users\\claud\\OneDrive\\CEFET\\2017-1 IC\\ARtigo\\Cod\\modelR.rda')

#############################################
#Ajuste dos dados para o dataset de teste
#############################################

df_teste_dummy$dia_2<-0
df_teste_dummy$dia_3<-0
df_teste_dummy$dia_4<-0
df_teste_dummy$dia_5<-0
df_teste_dummy$dia_6<-0
df_teste_dummy$dia_7<-0
df_teste_dummy$dia_8<-0
df_teste_dummy$dia_9<-0
df_teste_dummy$dia_10<-0
df_teste_dummy$dia_11<-0
df_teste_dummy$dia_17<-0
df_teste_dummy$dia_18<-0
df_teste_dummy$dia_24<-0
df_teste_dummy$dia_25<-0
df_teste_dummy$dia_26<-0
df_teste_dummy$dia_27<-0
df_teste_dummy$dia_28<-0
df_teste_dummy$dia_29<-0
df_teste_dummy$dia_30<-0
df_teste_dummy$dia_31<-0


df_teste_dummy <- df_teste_dummy[,c("dia_2","dia_3","dia_4" , "dia_5" , "dia_6" , "dia_7" , "dia_8" , "dia_9" , "dia_10" 
                                    ,"dia_11" , "dia_12" , "dia_13" , "dia_14" , "dia_15" , "dia_16" , "dia_17" , "dia_18" , "dia_19" 
                                    ,"dia_20" , "dia_21" , "dia_22" , "dia_23" , "dia_24" , "dia_25" , "dia_26" , "dia_27" , "dia_28" 
                                    ,"dia_29" , "dia_30" , "dia_31" , "hora_9" , "hora_10","hora_11","hora_12","hora_14","hora_15"
                                    ,"hora_16","hora_17","hora_18","hora_19","hora_20","hora_21","hora_22") ]
#############################################
#Predição da base de teste
#############################################


y_est<-predict(model,newxreg = df_teste_dummy)
names(y_est$pred)
names(df_teste_dummy)


#Exportação dos dados para arquivo - interação com o python
a=data.frame(y_est$pred)
write.table(a,file="c:\\temp\\armamodel.csv",sep=',')
#RMSE
sqrt(sum((df_teste$chamadas - y_est$pred)^2) / (140))

