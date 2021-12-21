import random
import math
import time
import itertools
import random
from random import randrange

salesman = traveling_salesman(num_cities=50)
possibleitr=int(15890700*95/100)


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

def Mutation(path):
    RandomIndex1 = random.randint(0, 5)
    RandomIndex2 = random.randint(0, 5)
    newPath=list(path[1])

    item1=newPath[RandomIndex1]

    newPath[RandomIndex1]=newPath[RandomIndex2]
    newPath[RandomIndex2]=item1

    return fitness(newPath),newPath

def do_Mutation(popsize,population,itr):
    child=population
    if popsize==0:
        return child
    i=0
    if(i < itr):
        rndn=random.randint(0,popsize-1)
        pop=population[rndn]
        temp=Mutation(pop)
        if pop[0]>temp[0]:
            child.append(temp)
            child.remove(pop)
        i=i+1
    
    return child
        
def Crossover(path1,path2):
    randIndex1=0
    randIndex2=0
    while True:
        randIndex1=random.randint(0,5)
        randIndex2=random.randint(0,5)
        if randIndex1 !=randIndex2:
            break
    first=min(randIndex1,randIndex2)
    last=max(randIndex1,randIndex2)
    child1=[]
    for i in range(first,last+1):
        child1.append(path1[1][i])
    child2=[]
    remain=6-len(child1)
    for item in path2[1]:
        if remain>0:
            if item not in child1:
                child2.append(item)
                remain=remain-1
        else:
            break
    newPath=child1+child2
    return fitness(newPath),newPath
        
        
def do_Crossover(popsize,population,itr) :
    child=[]
    
    i=0
    if(i < itr):
        pop1=population[random.randint(0,popsize-1)]
        pop2=population[random.randint(0,popsize-1)]
        temp=Crossover(pop1,pop2)
        if pop1[0]>temp[0]  or pop2[0]>temp[0]:
            child.append(temp)
        i=i+1
    return child

def GA(popsize,mutat_rate,cross_rate,itr):
    start=time.time()
    population=GenereatePopulation(popsize)
    #print(population[200])
    BestTour=[]
    steps=[]
    BestFitness=math.inf
    j=0
    cross_num=cross_rate*popsize
    mutat_num=mutat_rate*popsize
    print(cross_num)
    
    while j<itr:
        population.sort(key=lambda tup: tup[0])
        population=population[:popsize]
        j=j+1
        if population[0][0]<BestFitness:
            BestFitness=population[0][0]
            BestTour=population[0][1]
            steps.append(BestFitness)
            print("bestlength: "+str(BestFitness)+" best tour: "+str(BestTour)+" iteration: "+str(j))
        childc=do_Crossover(popsize,population,cross_num)
        childm=do_Mutation(len(childc),childc,mutat_num)
        population=population+childm
       
        
    return BestTour,BestFitness, itr , steps, time.time()-start

result=GA(1000,0.5,0.8,13000)
print('best_tour :',result[0],' best_length :', result[1], ' best_itr:', result[2],' best_time:',result[4])

salesman.new_tour(tour = result[0])
salesman.plot()

fvalues=result[3]
step=np.arange(0,len(fvalues))

plt.title('GA Search')
plt.plot(step, fvalues, '.-')
plt.xlabel('step')
plt.ylabel('length')

plt.show()


