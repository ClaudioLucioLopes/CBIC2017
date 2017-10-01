# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import operator
import math
import random


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import multiprocessing

##################Leitura dos dados
##TODO: Colocar este método fora
#df_serie = pd.read_csv('C:\\Users\\claud\\OneDrive\\CEFET\\2017-1 Com.Evol\\Trabalhos\\Artigo\\dados\\tab_9_22.csv',sep=';')
df_serie = pd.read_csv('C:\\temp\\featured_base.csv',sep=',', index_col=0)
#nome das colunas
list(df_serie.columns.values)


#df_serie =df_serie[0:100] 
#plot das serie
plt.plot(df_serie['chamadas'])
df_serie['chamadas'].describe()
#df_serie['chamadas'].hist()
lambda_cham = df_serie['chamadas'].mean()


random.seed(318)


pset = gp.PrimitiveSetTyped("main", [float,float, float,float,float, float,float,float, float, \
                                     float,float, float,float,float, float,float,float,float \
                                     , float,float,float,float,bool,bool,bool,bool,bool,bool \
                                     ,bool,bool,bool,bool,bool,bool,bool,bool], float)


# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
#pset.addPrimitive(protectedDiv, [float,float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

#pset.addPrimitive(operator.lt, [float, float], bool)
#pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addEphemeralConstant("coef1", lambda: random.uniform(-2, 2),float)
pset.addEphemeralConstant("coef2", lambda: random.randint(0, 1),float)
#pset.addEphemeralConstant("coef3", lambda: bool(random.getrandbits(1)),bool)
#pset.addEphemeralConstant("coef5", lambda: random.uniform(-2, 2),float)
#pset.addEphemeralConstant("coef6", lambda: random.uniform(-2, 2),float)
#pset.addEphemeralConstant("coef7", lambda: random.uniform(-2, 2),float)
##pset.addEphemeralConstant("rand01", lambda: random.random())
pset.addEphemeralConstant("lambdaT51_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda1_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD51_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT52_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda2_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD52_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT53_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda3_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD53_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT54_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda4_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD54_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT55_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda5_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD55_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT56_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda6_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD56_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT57_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda7_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD57_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT58_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda8_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD58_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT59_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda9_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD59_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)
pset.addEphemeralConstant("lambdaT510_cham", lambda: float(np.random.poisson(lambda_cham*5, 1)),float)
pset.addEphemeralConstant("lambda10_cham", lambda: float(np.random.poisson(lambda_cham, 1)),float)
pset.addEphemeralConstant("lambdaD510_cham", lambda: float(np.random.poisson(lambda_cham/5, 1)),float)


pset.renameArguments(ARG0='chamadas_mean')
pset.renameArguments(ARG1='chamadas_std')
pset.renameArguments(ARG2= 'Chamadas_hora_mean')
pset.renameArguments(ARG3='Chamadas_hora_std')
pset.renameArguments(ARG4='Chamadas_DiaSemana_mean')
pset.renameArguments(ARG5='Chamadas_DiaSemana_std')
pset.renameArguments(ARG6='Chamadas_dia_mean_x')
pset.renameArguments(ARG7='Chamadas_dia_std_x')
pset.renameArguments(ARG8='chamadas_q1')
pset.renameArguments(ARG9='chamadas_median')
pset.renameArguments(ARG10='chamadas_q3')
pset.renameArguments(ARG11='chamadas_lag1')
pset.renameArguments(ARG12='chamadas_lag2')
pset.renameArguments(ARG13='chamadas_lag3')
pset.renameArguments(ARG14='chamadas_lag4')
pset.renameArguments(ARG15='chamadas_lag5')
pset.renameArguments(ARG16='chamadas_lag6')
pset.renameArguments(ARG17='chamadas_lag7')
pset.renameArguments(ARG18='chamadas_lag8')
pset.renameArguments(ARG19='chamadas_lag9')
pset.renameArguments(ARG20='chamadas_lag10')
pset.renameArguments(ARG21='chamadas_lag11')
pset.renameArguments(ARG22= 'IND_hora_9')
pset.renameArguments(ARG23= 'IND_hora_10')
pset.renameArguments(ARG24= 'IND_hora_11')
pset.renameArguments(ARG25= 'IND_hora_12')
pset.renameArguments(ARG26= 'IND_hora_13')
pset.renameArguments(ARG27= 'IND_hora_14')
pset.renameArguments(ARG28= 'IND_hora_15')
pset.renameArguments(ARG29= 'IND_hora_16')
pset.renameArguments(ARG30= 'IND_hora_17')
pset.renameArguments(ARG31= 'IND_hora_18')
pset.renameArguments(ARG32= 'IND_hora_19')
pset.renameArguments(ARG33= 'IND_hora_20')
pset.renameArguments(ARG34= 'IND_hora_21')
pset.renameArguments(ARG35= 'IND_hora_22')

#pset.renameArguments(ARG3='IND_DiaSemana_1')
#pset.renameArguments(ARG4='IND_DiaSemana_2')
#pset.renameArguments(ARG5='IND_DiaSemana_3')
#pset.renameArguments(ARG6='IND_DiaSemana_4')
#pset.renameArguments(ARG7='IND_DiaSemana_5')



#
#pset.addTerminal(False, bool)
#pset.addTerminal(True, bool)

##Gera arvore completa 
#expr = gp.genFull(pset, min_=1, max_=5)
#tree = gp.PrimitiveTree(expr)
#print(tree)
#a = tree.from_string('sin(chamadas_lag4)',pset)
#
#points.apply(func,axis=1)
#func(row['chamadas_mean'],row['chamadas_std'],row['chamadas_lag1'],bool(row['IND_DiaSemana_1']),
#                          bool(row['IND_DiaSemana_2']),bool(row['IND_DiaSemana_3']),bool(row['IND_DiaSemana_4']),
#                          bool(row['IND_DiaSemana_5']))
#
##Apresenta a arvore gerada
#nodes, edges, labels = gp.graph(expr)
#import networkx as nx
#import matplotlib.pyplot as plt
#G=nx.Graph()
#G.add_nodes_from(nodes)
#G.add_edges_from(edges)
#nx.draw_networkx(G,labels=labels)

#######################Parte evolucionária
#função de miniminizaçãp
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#Criação do individuo do tipo arvore
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

#Funções de Criação dos individois do DEAP - Registros
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=3, max_=30)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)

#Funções de Criaçãodo processo de evoluçãodo DEAP -
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#Parametros da Evolução
#Avaliação da expressão - Função de Fitness  - usando erro ao quadrado
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
#    tree = gp.PrimitiveTree(individual)
#    print('--------------------------------')
#    print(tree)
    sqerrors=0
    for index, row in points.iterrows():
        sqerrors += (func(row['chamadas_mean'],row['chamadas_std'],row['Chamadas_hora_mean'], \
                          row['Chamadas_hora_std'],row['Chamadas_DiaSemana_mean'], \
                          row['Chamadas_DiaSemana_std'],row['Chamadas_dia_mean_x'],row['Chamadas_dia_std_x'], \
                          row['chamadas_q1'],row['chamadas_median'], \
                          row['chamadas_q3'],row['chamadas_lag1'],row['chamadas_lag2'], \
                          row['chamadas_lag3'],row['chamadas_lag4'],row['chamadas_lag5'], \
                          row['chamadas_lag6'],row['chamadas_lag7'],row['chamadas_lag8'], \
                          row['chamadas_lag9'],row['chamadas_lag10'],row['chamadas_lag11'], \
                          bool(row['IND_hora_9']),bool(row['IND_hora_10']),bool(row['IND_hora_11']), \
                          bool(row['IND_hora_12']),bool(row['IND_hora_13']),bool(row['IND_hora_14']), \
                          bool(row['IND_hora_15']),bool(row['IND_hora_16']),bool(row['IND_hora_17']), \
                          bool(row['IND_hora_18']),bool(row['IND_hora_19']),bool(row['IND_hora_20']), \
                          bool(row['IND_hora_21']),bool(row['IND_hora_22'])) - row['chamadas'])**2 
    return sqerrors / len(points),


toolbox.register("evaluate", evalSymbReg, points=df_serie.iloc[range(12,df_serie.shape[0])][['chamadas_mean','chamadas_std',
                          'Chamadas_hora_mean','Chamadas_hora_std','Chamadas_DiaSemana_mean', \
                          'Chamadas_DiaSemana_std','Chamadas_dia_mean_x','Chamadas_dia_std_x', \
                          'chamadas_q1','chamadas_median', \
                          'chamadas_q3','chamadas_lag1','chamadas_lag2', \
                          'chamadas_lag3','chamadas_lag4','chamadas_lag5', \
                          'chamadas_lag6','chamadas_lag7','chamadas_lag8', \
                          'chamadas_lag9','chamadas_lag10','chamadas_lag11',
                          'IND_hora_9','IND_hora_10','IND_hora_11', \
                          'IND_hora_12','IND_hora_13','IND_hora_14', \
                          'IND_hora_15','IND_hora_16','IND_hora_17', \
                          'IND_hora_18','IND_hora_19','IND_hora_20', \
                          'IND_hora_21','IND_hora_22','chamadas']])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))

def log_to_df(logbook):
    from functools import reduce
    from operator import add, itemgetter

    chapter_keys = logbook.chapters.keys()
    sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]

    data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                 in zip(sub_chaper_keys, logbook.chapters.values())]
    data = np.array([[*a, *b] for a, b in zip(*data)])

    columns = reduce(add, [["_".join([x, y]) for y in s] 
                       for x, s in zip(chapter_keys, sub_chaper_keys)])
    df = pd.DataFrame(data, columns=columns)

    keys = logbook[0].keys()
    data = [[d[k] for d in logbook] for k in keys]
    for d, k in zip(data, keys):
        df[k] = d
    return df
    
    
def execute_experiment():
    
    # Process Pool of 2 workers
    pool = multiprocessing.Pool(processes=2)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=100)
    #Elitism
    #https://groups.google.com/forum/#!searchin/deap-users/elitism%7Csort:relevance/deap-users/iannnLI2ncE/eI4BcVcwFwMJ
    #pop = tools.selBest(pop, int(0.1*len(pop))) + tools.selTournament(pop, len(pop)-int(0.1*len(pop)), tournsize=3)
    hof = tools.HallOfFame(1)

    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 400, stats=mstats,
                                   halloffame=hof, verbose=True)
    pool.close()
    # print log
    return pop, log, hof

def main():
    for i in range(5):
        print(i+1)
        pop, log, hof = execute_experiment()
        #Estatisticas da execução do experimento
        df_log_statistics = log_to_df(log)
        df_log_statistics.to_csv('c:\\temp\\estatisticas_'+str(i)+'_.csv')
        #Salva o hall of fame
        with open('c:\\temp\\HOF_'+str(i)+'_.pkl', 'wb') as pickle_file:
            pickle.dump(hof, pickle_file)
 
        
        
if __name__ == "__main__":
   import pickle
   main()
   for i in range(5):
       with open('c:\\temp\\HOF_'+str(i)+'_.pkl', 'rb') as input:
           hof = pickle.load(input)
       print(hof)
       print(hof[0].fitness)
       func = toolbox.compile(expr=hof[0])
       tree = gp.PrimitiveTree(hof[0])
       print(tree)
       y_est = list()
       for index, row in     df_serie.iloc[range(12,df_serie.shape[0])].iterrows():
           y_est.append(func(row['chamadas_mean'],row['chamadas_std'],row['Chamadas_hora_mean'], \
                       row['Chamadas_hora_std'],row['Chamadas_DiaSemana_mean'], \
                       row['Chamadas_DiaSemana_std'],row['Chamadas_dia_mean_x'],row['Chamadas_dia_std_x'], \
                       row['chamadas_q1'],row['chamadas_median'], \
                       row['chamadas_q3'],row['chamadas_lag1'],row['chamadas_lag2'], \
                       row['chamadas_lag3'],row['chamadas_lag4'],row['chamadas_lag5'], \
                       row['chamadas_lag6'],row['chamadas_lag7'],row['chamadas_lag8'], \
                       row['chamadas_lag9'],row['chamadas_lag10'],row['chamadas_lag11'], \
                       bool(row['IND_hora_9']),bool(row['IND_hora_10']),bool(row['IND_hora_11']), \
                       bool(row['IND_hora_12']),bool(row['IND_hora_13']),bool(row['IND_hora_14']), \
                       bool(row['IND_hora_15']),bool(row['IND_hora_16']),bool(row['IND_hora_17']), \
                       bool(row['IND_hora_18']),bool(row['IND_hora_19']),bool(row['IND_hora_20']), \
                       bool(row['IND_hora_21']),bool(row['IND_hora_22'])))
        
       plt.plot(df_serie['chamadas'])
       plt.plot(y_est)
       plt.show()
       #Calculo do RMSE
       sum((y_est- np.array(df_serie.iloc[range(12,df_serie.shape[0])]['chamadas']))**2) \
                        / df_serie.iloc[range(12,df_serie.shape[0])].shape[0]
    


                                        
#801.638                                     
#if_then_else(IND_hora_9, if_then_else(IND_hora_13, 463.0, Chamadas_hora_mean), add(sub(chamadas_lag1, sin(add(neg(chamadas_median), neg(add(Chamadas_hora_mean, mul(mul(490.0, if_then_else(and_(not_(IND_hora_14), and_(not_(and_(not_(IND_hora_13), IND_hora_20)), not_(or_(IND_hora_15, IND_hora_15)))), 108.0, 14.0)), cos(add(Chamadas_hora_mean, 490.0)))))))), if_then_else(IND_hora_17, neg(23.0), cos(mul(mul(add(490.0, neg(if_then_else(eq(if_then_else(not_(not_(and_(IND_hora_17, eq(add(17.0, Chamadas_dia_std_x), sub(add(486.0, cos(519.0)), mul(493.0, 120.0)))))), Chamadas_hora_mean, 493.0), if_then_else(IND_hora_14, Chamadas_hora_mean, 495.0)), cos(sin(if_then_else(IND_hora_15, sub(Chamadas_dia_mean_x, 73.0), Chamadas_hora_mean))), sin(if_then_else(IND_hora_15, sub(Chamadas_dia_mean_x, 73.0), Chamadas_hora_mean))))), if_then_else(or_(or_(eq(sin(add(add(17.0, Chamadas_dia_std_x), neg(neg(Chamadas_hora_mean)))), 455.0), and_(or_(and_(not_(not_(IND_hora_9)), IND_hora_14), and_(not_(IND_hora_16), or_(or_(not_(IND_hora_14), IND_hora_17), not_(IND_hora_22)))), IND_hora_14)), eq(chamadas_lag2, chamadas_median)), 108.0, if_then_else(eq(add(add(Chamadas_hora_mean, if_then_else(IND_hora_13, 108.0, neg(sin(add(mul(22.0, 97.0), add(Chamadas_DiaSemana_mean, Chamadas_hora_mean)))))), if_then_else(and_(not_(not_(and_(IND_hora_9, IND_hora_20))), not_(or_(eq(if_then_else(not_(not_(and_(IND_hora_17, IND_hora_20))), sin(neg(chamadas_lag4)), 73.0), 455.0), and_(not_(IND_hora_9), not_(not_(IND_hora_13)))))), mul(mul(chamadas_mean, 14.0), mul(455.0, if_then_else(IND_hora_11, if_then_else(eq(chamadas_lag7, chamadas_lag9), cos(86.0), cos(18.0)), chamadas_lag6))), add(mul(add(if_then_else(IND_hora_18, chamadas_lag6, 93.0), neg(chamadas_lag7)), cos(add(16.0, Chamadas_DiaSemana_mean))), mul(sin(if_then_else(IND_hora_18, 18.0, 100.0)), sin(mul(529.0, chamadas_lag3)))))), Chamadas_dia_mean_x), 108.0, 26.0))), cos(add(Chamadas_hora_mean, mul(mul(490.0, if_then_else(and_(not_(eq(if_then_else(not_(IND_hora_9), Chamadas_hora_mean, 495.0), if_then_else(IND_hora_14, Chamadas_hora_mean, add(sin(if_then_else(IND_hora_20, chamadas_lag6, neg(chamadas_median))), mul(neg(chamadas_lag8), neg(Chamadas_hora_mean)))))), and_(not_(IND_hora_14), and_(not_(IND_hora_14), or_(IND_hora_11, and_(not_(IND_hora_14), and_(and_(not_(IND_hora_13), and_(not_(IND_hora_9), not_(IND_hora_13))), not_(IND_hora_13))))))), 108.0, mul(if_then_else(not_(IND_hora_13), mul(if_then_else(not_(or_(eq(mul(if_then_else(IND_hora_11, chamadas_lag3, 107.0), sin(chamadas_lag7)), sin(cos(chamadas_mean))), and_(and_(not_(and_(IND_hora_9, IND_hora_20)), eq(sub(24.0, Chamadas_hora_mean), add(chamadas_mean, chamadas_lag3))), and_(and_(eq(Chamadas_dia_mean_x, chamadas_lag4), not_(IND_hora_13)), or_(eq(Chamadas_hora_mean, 25.0), eq(Chamadas_DiaSemana_std, chamadas_q1)))))), if_then_else(IND_hora_15, sub(Chamadas_dia_mean_x, 73.0), Chamadas_hora_mean), 495.0), chamadas_mean), 495.0), chamadas_mean))), cos(add(Chamadas_hora_mean, 490.0))))))))))