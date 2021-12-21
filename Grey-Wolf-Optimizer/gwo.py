#Grey Wolf Optimizer
from __future__ import division
import numpy as np
import math
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#--- COST FUNCTION ------------------------------------------------------------+

# Fitness function
def Fitness(z_arr):
    if len(z_arr.shape)==2:
        return max(z_arr.max(axis=0))
    elif len(z_arr.shape)==1:
        return max(z_arr)
    elif len(z_arr.shape)==0:
        return z_arr
    
def calculate_Fitness_GWO(population,type_p):
    fit=[]
    
    for pop in population:

        if type_p==1 or type_p==2:
            now=Fitness(pop[2])
            fit.append(now)
        else:
            now=Fitness(pop[2])
            fit.append(now)

    return fit  

def create_population_GWO(num,type_p,row,column,problem):
    population=[]
    x_arr=0
    y_arr=0
    for i in range(0,num):
        if type_p==1:
            x_arr=np.array(random.uniform(-50, 50))
            y_arr=np.array(random.uniform(-50, 50))
            
        elif type_p==2:
            x_arr =np.array([random.uniform(-50, 50) for p in range(0,column)])
            y_arr =np.array([random.uniform(-50, 50) for p in range(0,column)])

        elif type_p==3:
            x_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    x_arr[i][j]=random.uniform(-50,50)
            y_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    y_arr[i][j]=random.uniform(-50,50)
        if problem.__name__=="problem_3":
            population.append([x_arr,y_arr,problem(x_arr,y_arr,x_arr,y_arr)])
        else:
            population.append([x_arr,y_arr,problem(x_arr,y_arr)])
            
    return population
def Calculate_Distance(wolf,prey,a):
    r1=random.random()
    r2=random.random()
    A1=2*a*r1-a;
    C1=2*r2; 
                
    D=abs(C1*wolf-prey); # Equation (3.5)-part 1
    X=wolf-A1*D;
    return X
def update_population(type_p,problem,alpha,beta,delta,population,a):

    x_arr=0
    y_arr=0
    for pop in population:
        
        if type_p==1:
            
            pop[0]=(Calculate_Distance(alpha[0],pop[0],a)+Calculate_Distance(beta[0],pop[0],a)+
                    Calculate_Distance(delta[0],pop[0],a))/3
            pop[1]=(Calculate_Distance(alpha[1],pop[1],a)+Calculate_Distance(beta[1],pop[1],a)
                    +Calculate_Distance(delta[1],pop[1],a))/3           
           
        if type_p==2:
            for j in range (0,len(pop[0])):
                pop[0][j]=(Calculate_Distance(alpha[0][j],pop[0][j],a)+
                           Calculate_Distance(beta[0][j],pop[0][j],a)+Calculate_Distance(delta[0][j],pop[0][j],a))/3
                pop[1][j]=(Calculate_Distance(alpha[1][j],pop[1][j],a)+Calculate_Distance(beta[1][j],pop[1][j],a)+
                           Calculate_Distance(delta[1][j],pop[1][j],a))/3 
        
            
        if type_p==3:
            s=pop[0].shape
  
            for m in range(0,s[0]):
                for n in range(0, s[1]):
                      pop[0][m][n]=(Calculate_Distance(alpha[0][m][n],pop[0][m][n],a)
                                    +Calculate_Distance(beta[0][m][n],pop[0][m][n],a)
                                    +Calculate_Distance(delta[0][m][n],pop[0][m][n],a))/3
                      pop[1][m][n]=(Calculate_Distance(alpha[1][m][n],pop[1][m][n],a)
                                    +Calculate_Distance(beta[1][m][n],pop[1][m][n],a)
                                    +Calculate_Distance(delta[1][m][n],pop[1][m][n],a))/3 
                    
        if problem.__name__=="problem_3":          
            pop[2]=problem(pop[0],pop[1],pop[0],pop[1])
        else:
            pop[2]=problem(pop[0],pop[1])

    return population

def GWO(itr, population,Num_pop,problem ):
    Alpha_pos=0
    Alpha_score=math.inf
    
    Beta_pos=0
    Beta_score=math.inf
    
    Delta_pos=0
    Delta_score=math.inf

    sequence=[]
    fitness=[math.inf]*Num_pop
    
    k=population[0][2].shape
    Type=1
    if len(k)==1 :
        Type=2
    elif len(k)==2 :
        Type=3
     

    iteration=0
    for i in range(0,itr):
        #print(population)
        fitness=calculate_Fitness_GWO(population,Type)
        indices=np.argsort(fitness)
        
        if fitness[indices[0]] < Alpha_score:
            iteration=i
            Alpha_pos=population[indices[0]].copy()
            Alpha_score=fitness[indices[0]]
            sequence.append(Alpha_score)

        if fitness[indices[1]]< Beta_score:
            Beta_pos=population[indices[1]].copy()
            Beta_score=fitness[indices[1]]
            
            
        if fitness[indices[2]]< Delta_score:
            Delta_pos=population[indices[2]].copy()
            Delta_score=fitness[indices[2]]
            
        a=2-i*((2)/itr)
        population=update_population(Type,problem,Alpha_pos,Beta_pos,Delta_pos,population,a)
    
    return Alpha_score,iteration,sequence,Alpha_pos[0],Alpha_pos[1],Alpha_pos[2]

for s in range(1,5):
    np.random.seed(s)
    print("seed number: "+str(s))
    
    print("problem_1 : ")
    population=create_population_GWO(40,1,0,0,problem_1)

    result= GWO(1000, population,40,problem_1 )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('GWO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show()
    
    print("problem_2 : ")
    population=create_population_GWO(40,2,0,2,problem_2)

    result= GWO(1000, population,40,problem_2 )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('GWO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show()
    
    
    print("problem_3 : ")
    population=create_population_GWO(40,3,2,2,problem_3)

    result= GWO(1000, population,40,problem_3 )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('GWO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show()