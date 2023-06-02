from cProfile import label
import re
import time
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for projection='3d'
from sklearn.manifold import TSNE
from sklearn import preprocessing


from collections import Counter
import copy

import os
from datetime import datetime
import yaml
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(1)


class Evolution():
    def __init__(self):
        now = datetime.now()  # 获得当前时间
        timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.dir = '../data/GB1/dataset/raw/' + timestr + "/"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        self.data_path = '../data/GB1/dataset/raw/GB1.csv'
        self.all_data =pd.read_csv(self.data_path)
        self.init_data = pd.read_csv(self.data_path)
        self.name_length = len(self.init_data['resn'][0]) # 4
        
        name = np.array(self.init_data['resn'])
        fitness = np.array(self.init_data['fitness'], dtype=float)
        self.population = np.array(list(zip(name, fitness)))
        
        self.n_generation = 300
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        
        self.max_fitness = max(fitness)
        ind = np.argmax(fitness)
        self.max_name = name[ind]
        
        print(f"max fitness: {self.max_name, self.max_fitness}")
        print(f'fitness distribution: {max(fitness), min(fitness), np.mean(fitness), np.median(fitness), np.percentile(fitness, 97)}')
        plt.plot(sorted(fitness))
        plt.savefig('GB1_fitness.png')
        plt.close()

        self.size = len(name)
        
        self.kids_size = self.size
        self.kids = np.empty_like(self.population)
        
        self.all_20_p = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    def get_fitness(self, kid):
        
        if kid in self.init_data['resn']:
            fitness = float(self.init_data[self.init_data['resn']==kid]['fitness'])
        else:
            fitness = 0
        return fitness
    
    def make_kid(self):
        for i in range(self.size):
            coin = np.random.uniform()
            if coin < self.crossover_rate:
                p1, p2 = np.random.choice(np.arange(self.size), size=2, replace=False)
                cp = np.random.randint(0, self.name_length)  # crossover points
                name1 = self.population[p1][0]
                name2 = self.population[p2][0]
            
                kid = name1[:cp]+name2[cp:]
                
            else:
                ind = np.random.choice(np.arange(self.size), replace=False)
                kid = self.population[ind][0]
                
                
            if np.random.uniform() < self.mutation_rate:
                # mutate
                
                mp = np.random.randint(0, self.name_length)
                if mp == 0:
                    kid = np.random.choice(self.all_20_p) + kid[mp+1:]
                elif mp == 3:
                    kid = kid[:mp] + np.random.choice(self.all_20_p)
                else:
                    kid = kid[:mp] + np.random.choice(self.all_20_p) + kid[mp+1:]
                
            fitness = self.get_fitness(kid)
            name = kid
            self.kids[i] = (name, fitness)
            
    def select(self):
        
        all_population = np.concatenate((self.population, self.kids))
        sort_pop = sorted(all_population, key=lambda d: d[1], reverse=True) # sort by fitness
        fitness = np.array([float(v[1]) for v in sort_pop], dtype = float)
        # median = np.median(fitness)
        key_point = min(np.percentile(fitness, 0.97), 0.8)
        print(key_point)
        new_pop = np.empty_like(self.population)
        
        ind = 0
        sign = True
        while sign:
            if ind:
                print(f'current population size: {ind}, not enough, select again.')
            for tuple in sort_pop:
                # delta = float(tuple[1])-key_point if float(tuple[1]) > key_point else (float(tuple[1])-key_point)*3
                delta = (float(tuple[1])-key_point)*3
                p = 1/ (1+ np.exp(delta)) # probability of dropping
                
                coin = np.random.uniform()
                if coin < p: # drop
                    pass
                else: # keep
                    new_pop[ind] = copy.deepcopy(tuple)
                    ind += 1
                
                if ind >= self.size:
                    sign = False
                    break
                
        self.population = new_pop
        
    
        
        
    def evo_run(self):        
        sign = True
        for i in range(self.n_generation):
            print(f'Generation {i}' + 20* '*')
            self.make_kid()
            self.select()
            print("New population generated!")
            
            name = [v[0] for v in self.population]
            if name.count(self.max_name) >= int(0.5*self.size):
                sign=False
                print("Finished due to max fitness occupation.")
                break
            if (i +1) % 10 == 0 and i!=self.n_generation-1: # save 
                self.save_frequency(postfix=str(i+1))
                
        if sign:
            print('Finished normally.')
        
    def get_frequency(self):
        name = [v[0] for v in self.population]
        return Counter(name)
    
    def save_frequency(self, postfix=''):
        frequency_dict = evo.get_frequency()
        df = pd.DataFrame({'resn': frequency_dict.keys(), 'freq':frequency_dict.values()})
        # df.to_csv('../data/GB1/dataset/raw/' + 'GB1_freq' + postfix + '.csv')
        
        tmp = df.merge(self.all_data, on='resn')
        tmp = tmp[['resn', 'freq', 'fitness','label']]
        tmp.sort_values('fitness', inplace=True, ascending=False)
      
        file_name = 'GB1_freq' + postfix + '.csv'
        tmp.to_csv(self.dir + file_name)
        
    
    
if __name__  == '__main__':
        
        evo = Evolution()
        print('start runing')
        evo.evo_run()
        evo.save_frequency()
        
        
        
        
        
        
        