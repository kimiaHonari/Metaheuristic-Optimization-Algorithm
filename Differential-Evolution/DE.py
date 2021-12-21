import random
import math
import time
import itertools
import random
from random import randrange

salesman = traveling_salesman(num_cities=50)

def fitness(pop):

    salesman.new_tour(tour = pop)
    cost=salesman.tour_length()
    return cost

def GenereatePopulation(popsize):
    pop=[]
    itr=0
    j=0
    for i in range(0,int(50/6)):
        path=list(range(j,j+6))
        p=fitness(path),path
        pop.append(p)
        j=j+6
    path=[48,49,0,1,2,3]
    p=fitness(path),path
    pop.append(p)
    
    popsize=popsize-9
    
    for t in range(0,popsize):
        rand=randrange(0,len(pop))
        path=random.sample(pop[rand][1], len(pop[rand][1]))
        p=fitness(path),path
        pop.append(p)
        #itr=itr+1
        #if(itr==size):
            #return pop
    return pop

def do_DE(popsize,population,mutat_rate,recombination):
    child=[]
    for j in range(0, popsize):
        candidates=list(range(0,popsize))
        candidates.remove(j)
        random_index=random.sample(candidates,3)
        
        p1=population[random_index[0]]
        p2=population[random_index[1]]
        p3=population[random_index[2]]
        p4=population[j]
        
        #compute diff
        v=[]
        for d in range(0,6):
            v.append(int((p2[1][d]-p3[1][d])*mutat_rate)+p1[1][d])
            if v[d]>49:
                v[d]=49
            if v[d]<0:
                v[d]=0
        #end compute diff
        
        #recombination
        Newpath=[]
        NF=0
        j=0;
        itr=0
        while True:
            if itr<6:
                if random.random()<= recombination:
                    if v[itr] not in Newpath:
                        Newpath.append(v[itr])
                        j=j+1
                else:
                
                    if p4[1][itr] not in Newpath:
                        Newpath.append(p4[1][itr])
                        j=j+1
            itr=itr+1
            #print(Newpath)
            if itr>=6 and j<6:
                for item in p4[1]:
                    if item not in Newpath:
                        Newpath.append(item)
                        j=j+1
                        if j==6:
                            break
            if j==6:
                break
        #end recombination
            
        
        #compute the fitness for a child and if it has a better fitness than the parent, append a child to a population
        
        fit=fitness(Newpath)
        
        if(fit < p4[0]):
            path= fit,Newpath
            child.append(path)

    return child
        
        
    
def DE(popsize,mutat_rate,recombination,itr):
    start=time.time()
    population=GenereatePopulation(popsize)
    #print(population[200])
    BestTour=[]
    steps=[]
    BestFitness=math.inf
    j=0


    while j<itr:
        population.sort(key=lambda tup: tup[0])
        population=population[:popsize]
        if population[0][0]<BestFitness:
            BestFitness=population[0][0]
            BestTour=population[0][1]
            steps.append(BestFitness)
            print("bestlength: "+str(BestFitness)+" best tour: "+str(BestTour)+"iteartion: "+str(j))
        child=do_DE(popsize,population,mutat_rate,recombination)
        
        population=population+child
        j=j+1
        
        

            
    return BestTour,BestFitness, itr , steps, time.time()-start

result=DE(900,0.6,0.3,500)
print('best_tour :',result[0],' best_length :', result[1], ' best_itr:', result[2],' best_time:',result[4])

salesman.new_tour(tour = result[0])
salesman.plot()

fvalues=result[3]
step=np.arange(0,len(fvalues))

plt.title('DE Search')
plt.plot(step, fvalues, '.-')
plt.xlabel('step')
plt.ylabel('length')

plt.show()
