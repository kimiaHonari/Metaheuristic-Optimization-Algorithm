#Particle Swarm optimization
from __future__ import division
import numpy as np
import math
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Blackbox_Problems_fixed import problem_1, problem_2, problem_3
#--- COST FUNCTION ------------------------------------------------------------+

# Fitness function
def Fitness(z_arr):
    if len(z_arr.shape)==2:
        return max(z_arr.max(axis=0))
    elif len(z_arr.shape)==1:
        return max(z_arr)
    elif len(z_arr.shape)==0:
        return z_arr
    
def calculate_Fitness(population,Best_positions,fitness,type_p):
    fit=[]
    i=0
    
    for pop in population:

        if type_p==1 or type_p==2:
            now=Fitness(pop[2])
            pre=fitness[i]
            if pre >now:
                Best_positions[i]=pop
            fit.append(now)
        else:
            now=Fitness(pop[2])
            pre=fitness[i]
            if pre >now:
                Best_positions[i]=pop
            fit.append(now)
        i=i+1
    return fit,Best_positions
                            
                            
def create_population(num,type_p,row,column,problem):
    population=[]
    x_arr=0
    y_arr=0
    for i in range(0,num):
        if type_p==1:
            x_arr=np.array(random.uniform(-10, 10))
            y_arr=np.array(random.uniform(-10, 10))
            
        elif type_p==2:
            x_arr =np.array([random.uniform(-10, 10) for p in range(0,column)])
            y_arr =np.array([random.uniform(-10, 10) for p in range(0,column)])

        elif type_p==3:
            x_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    x_arr[i][j]=random.uniform(-10,10)
            y_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    y_arr[i][j]=random.uniform(-10,10)
        if problem.__name__=="problem_3":
            population.append([x_arr,y_arr,problem(x_arr,y_arr,x_arr,y_arr)])
        else:
            population.append([x_arr,y_arr,problem(x_arr,y_arr)])
            
    return population
                            
                            
def create_velocity(num,type_p,row,column):
    velocity=[]
    for i in range(0,num):
        if type_p==1:
            x=random.uniform(-1, 1)
            y=random.uniform(-1, 1)
            velocity.append([x,y])
        elif type_p==2:
            x_arr =np.array([random.uniform(-1, 1) for i in range(0,column)])
            y_arr =np.array([random.uniform(-1, 1) for i in range(0,column)])
            velocity.append([x_arr,y_arr])
        elif type_p==3:
            x_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    x_arr[i][j]=random.uniform(-1,1)
            y_arr = np.zeros(shape=(row,column))
            for i in range(0,row):
                for j in range(0,column):
                    y_arr[i][j]=random.uniform(-1,1)
            velocity.append([x_arr,y_arr])
    return velocity
                            
                            
                            
def update_velocity_population  (problem,c1,c2,w,best_position,global_best_position_x,global_best_position_y,population,velocity,type_p):
    
    i=0
   
    for v in velocity:

        if type_p==1:
            r1=random.random()
            r2=random.random()
            vel_cognitive=c1*r1*(best_position[i][0]-population[i][0])
            vel_social=c2*r2*(global_best_position_x-population[i][0])
            v[0]=w*v[0]+vel_cognitive+vel_social
            population[i][0]=population[i][0]+v[0]

            vel_cognitive=c1*r1*(best_position[i][1]-population[i][1])
            vel_social=c2*r2*(global_best_position_y-population[i][1])
            v[1]=w*v[1]+vel_cognitive+vel_social
            population[i][1]=population[i][1]+v[1]
            
           
        if type_p==2:

            for j in range (0,len(v[0])):
                r1=random.random()
                r2=random.random()

                vel_cognitive=c1*r1*(best_position[i][0][j]-population[i][0][j])
                vel_social=c2*r2*(global_best_position_x[j]-population[i][0][j])
                v[0][j]=w*v[0][j]+vel_cognitive+vel_social
                
                population[i][0][j]=population[i][0][j]+v[0][j]  
         
                vel_cognitive=c1*r1*(best_position[i][1][j]-population[i][1][j])
                vel_social=c2*r2*(global_best_position_y[j]-population[i][1][j])
                v[1][j]=w*v[1][j]+vel_cognitive+vel_social
                population[i][1][j]=population[i][1][j]+v[1][j] 
                
            
            
        if type_p==3:
            s=v[0].shape
  
            for m in range(0,s[0]):
                for n in range(0, s[1]):
                    r1=random.random()
                    r2=random.random()
                    vel_cognitive=c1*r1*(best_position[i][0][m][n]-population[i][0][m][n])
                    vel_social=c2*r2*(global_best_position_x[m][n]-population[i][0][m][n])
                    v[0][m][n]=w*v[0][m][n]+vel_cognitive+vel_social
                    population[i][0][m][n]=population[i][0][m][n]+v[0][m][n]
                            
                    vel_cognitive=c1*r1*(best_position[i][1][m][n]-population[i][1][m][n])
                    vel_social=c2*r2*(global_best_position_y[m][n]-population[i][1][m][n])
                    v[1][m][n]=w*v[1][m][n]+vel_cognitive+vel_social 
                    population[i][1][m][n]=population[i][1][m][n]+v[1][m][n]
                    
        if problem.__name__=="problem_3":          
            population[i][2]=problem(population[i][0],population[i][1],population[i][0],population[i][1])
        else:
            population[i][2]=problem(population[i][0],population[i][1])

        i=i+1
                            
    return velocity, population

                            
def PSO(itr,Num_pop,problem,population,velocity,w=0.5,c1=1,c2=2 ):

    Best_Value=math.inf
    sequence=[]
    Best_z=0
    Best_x=0
    Best_y=0
    Best_positions=population
    fitness=[math.inf]*Num_pop
    k=population[0][2].shape
    Type=1
    if len(k)==1 :
        Type=2
    elif len(k)==2 :
        Type=3
     

    iteration=0
    for i in range(0,itr):

        fitness,Best_positions=calculate_Fitness(population,Best_positions,fitness,Type)

        indices=np.argsort(fitness)
   
        best_indx=indices[0]

        if fitness[best_indx]< Best_Value:
            Best_Value=fitness[best_indx]
            Best_x=population[best_indx][0]
            Best_y=population[best_indx][1]
            Best_z=population[best_indx][2]
            sequence.append(Best_Value)
            iteration=i

        velocity,population=update_velocity_population (problem,c1,c2,w,Best_positions,Best_x,Best_y,population,velocity,Type)
    
    return Best_Value,iteration,sequence,Best_x,Best_y,Best_z

for s in range(1,5):
    np.random.seed(s)
    print("seed number: "+str(s))

        
    print("problem_1 : ")
    population=create_population(40,1,0,0,problem_1)

    velocity=create_velocity(40,1,0,0)
    result= PSO(1000,40,problem_1,population,velocity,0.01,0.5,1  )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('PSO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show()
      
    print("problem_2 : ")
    population=create_population(40,2,0,2,problem_2)
    velocity=create_velocity(40,2,0,2)
    result= PSO(1000,40,problem_2,population,velocity,0.01,0.5,1 )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('PSO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show() 
        
          
    print("problem_3 : ")
    population=create_population(40,3,2,2,problem_3)
    velocity=create_velocity(40,3,2,2)

    result= PSO(1000,40,problem_3,population,velocity,0.01,0.5,1  )
    print('best_value :',result[0],' iteration :', result[1], " Best_x :",result[3], " Best_y :",result[4]," Best_z :",result[5])
    fvalues=result[2]
    step=np.arange(0,len(fvalues))

    plt.title('PSO Search')
    plt.plot(step, fvalues, '.-')
    plt.xlabel('step')
    plt.ylabel('length')

    plt.show() 
