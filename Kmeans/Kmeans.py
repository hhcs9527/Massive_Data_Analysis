import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[8]").setAppName("hw3")
sc = SparkContext(conf=conf).getOrCreate()

centroid = []
k = 0
countsByKey = {}


def centroid_loader(path):
    global centroid, k
    centroid  = sc.textFile(path).map(lambda x : list(map(float, x.split(' ')))).collect()
    k = len(centroid)


def read_file():
    data  = sc.textFile('data.txt').map(lambda x : tuple(list(map(float, x.split(' ')))) )
    return data


def calculate_Euclidean_distance(input_list):
    global centroid, k
    distance2centroid = []
    node_cluster = []
        
    for i in range(k):
        c = [(a_i - b_i)*(a_i - b_i) for a_i, b_i in zip(input_list, centroid[i])]
        distance2centroid.append(math.sqrt(sum(c)))
        
    node_cluster.append([input_list, distance2centroid])
        
    return node_cluster


def calculate_Manhattan_distance(input_list):
    global centroid, k
    distance2centroid = []
    node_cluster = []
        
    for i in range(k):
        c = [abs(a_i - b_i) for a_i, b_i in zip(input_list, centroid[i])]
        distance2centroid.append(sum(c))
        
    node_cluster.append([input_list, distance2centroid])
        
    return node_cluster


def Assign2new_centroid(input_data):
    # find the minimum distance to each centroid
    # Assign to new cluster
    # (cluster, node)
    # value, key, then reduce by key & average, you will get new centroid
    key_value = input_data.map(lambda x : (x[0][1].index(min(x[0][1])), x[0][0]) )
    
    return key_value
    

def calculate_Euclidean_cost(input_list):
    # distance where we find the new centroid
    # input_list format -> (cluster, node)
    return(sum([(a_i - b_i)*(a_i - b_i) for a_i, b_i in zip(input_list[1], centroid[input_list[0]])]))


def calculate_Manhattan_cost(input_list):
    # distance where we find the new centroid
    # input_list format -> (cluster, node)
    return(sum([abs(a_i - b_i) for a_i, b_i in zip(input_list[1], centroid[input_list[0]])]))    
    
    
def average(input_list):
    global countsByKey
    #(cluster, node)
    coordinate = []
    for i in input_list[1]:
        i /= countsByKey[input_list[0]]
    
    #print(input_list)
    coordinate = input_list[1]
    return coordinate 
    
    
def add(a, b):
    return [a_i + b_i for a_i, b_i in zip(a, b)]


def find_new_centroid(key_value):
    # Average the sum of the node coordinate
    new = []
    for i in range(len(key_value[1])):
        new.append(key_value[1][i]/countsByKey[key_value[0]])
    
    # Replace old centroid with new centroid 
    return new


def Mapper_Euclidean(data):
    # calculate_Euclidean_distance
    input_data = data.map(calculate_Euclidean_distance)
        
    # Assign to new centroid 
    key_value = Assign2new_centroid(input_data)
    
    return key_value


def Mapper_Manhattan(data):
    # calculate_Euclidean_distance
    input_data = data.map(calculate_Manhattan_distance)
        
    # Assign to new centroid 
    key_value = Assign2new_centroid(input_data)
    
    return key_value


def Reducer(key_value):
    global countsByKey, centroid
    # find_new_centroid
    # find the number of value for each key
    countsByKey = key_value.countByKey()
    # accumulate same key value from same cluster
    key_value = key_value.reduceByKey(lambda x,y : add(x,y))

    key_value = key_value.map(find_new_centroid)
    
    #update centroid
    centroid = key_value.collect()
    
    return centroid


def K_mean_run_Euclidean(iter, path):
    global centroid
    data = read_file()
    centroid_loader(path)
    cost_list = []
    for i in range(iter):

        key_value = Mapper_Euclidean(data)
        # calculate_Euclidean_cost
        cost_list.append(sum(key_value.map(calculate_Euclidean_cost).collect()))

        centroid = Reducer(key_value)

    return centroid, cost_list
    
    
def K_mean_run_Manhattan(iter, path):
    data = read_file()
    centroid_loader(path)
    cost_list = []
    global centroid, countsByKey
    for i in range(iter):
        
        key_value = Mapper_Manhattan(data)
        
        # calculate_Euclidean_cost
        cost_list.append(sum(key_value.map(calculate_Manhattan_cost).collect()))
        
        centroid = Reducer(key_value)
    
    return centroid, cost_list

    
        


class Compare():
    def __init__(self, iter, EU):
        self.iter = iter       
        self.EU = EU
        if self.EU:
            self.c1_centroid, self.c1_costlist = K_mean_run_Euclidean(self.iter,'c1.txt')
            self.c2_centroid, self.c2_costlist = K_mean_run_Euclidean(self.iter, 'c2.txt')
        else:
            self.c1_centroid, self.c1_costlist = K_mean_run_Manhattan(self.iter,'c1.txt')
            self.c2_centroid, self.c2_costlist = K_mean_run_Manhattan(self.iter, 'c2.txt')
            
            
    def compute_Euclidean(self, object):
        res = np.zeros((10,10))
        for i in range(10):
            for j in range(i,10):
                res[i,j] = np.sqrt(np.sum(( np.array(object[i])-np.array(object[j]) )**2))
        res_Euclidean = pd.DataFrame(res)
        return res_Euclidean
        
        
    def compute_Manhattan(self, object):
        res = np.zeros((10,10))
        for i in range(10):
            for j in range(i,10):
                res[i,j] = np.sum(np.abs(np.array(object[i])-np.array(object[j])))
        res_Manhattan = pd.DataFrame(res)
        return res_Manhattan        
    
    
    def compute_Percentage_improvement(self):
        improvement_c1 = abs(self.c1_costlist[-1] - self.c1_costlist[0]) / self.c1_costlist[0]
        improvement_c2 = abs(self.c2_costlist[-1] - self.c2_costlist[0]) / self.c2_costlist[0]
        
        if self.EU:
            print("Using Euclidean with c1 as inital centroid improves {:2.2f} %".format(improvement_c1))
            print("Using Euclidean with c2 as inital centroid improves {:2.2f} %".format(improvement_c2))
        else:
            print("Using Manhattan with c1 as inital centroid improves {:2.2f} %".format(improvement_c1))
            print("Using Manhattan with c2 as inital centroid improves {:2.2f} %".format(improvement_c2))


    def distance_compare(self):
        # Comparision with 2 measurement
        c1_EU = self.compute_Euclidean(self.c1_centroid)
        c1_MA = self.compute_Manhattan(self.c1_centroid)
        c2_EU = self.compute_Euclidean(self.c2_centroid)
        c2_MA = self.compute_Manhattan(self.c2_centroid)
        
        return c1_EU, c1_MA, c2_EU, c2_MA
    
    
    def plot_comp(self):
        c = [i for i in range(1,self.iter+1)]

        plt.plot(c, self.c1_costlist, label = 'c1_cost')
        plt.plot(c, self.c2_costlist, label = 'c2_cost')
        
        # show x range(刻度)
        plt.xticks(range(1,self.iter+1))
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.legend(loc='upper right')
        
        if self.EU:
            plt.title('Compare_Euclidean')
            plt.savefig('Compare_Euclidean.png')
        else:
            plt.title('Compare_Manhattan')
            plt.savefig('Compare_Manhattan.png')
        
        # close fig, so that wouldn't plot on it repeatly
        plt.clf()
            
        
    def do_homework(self):
        self.compute_Percentage_improvement()
        self.plot_comp() 
        return self.distance_compare()


        
if __name__ == "__main__":
    st = time.time()

    #K_mean_run_Euclidean(5,'c1.txt')
    #K_mean_run_Manhattan(5,'c2.txt')
    # Euclidean part
    C_EU = Compare(20, True)
    EU_c1_EU, EU_c1_MA, EU_c2_EU, EU_c2_MA = C_EU.do_homework()
    
    # Manhattan part
    C_MA = Compare(20, False)
    MA_c1_EU, MA_c1_MA, MA_c2_EU, MA_c2_MA = C_MA.do_homework()
    
    end = time.time()
    print('spend : ', end-st)
    sc.stop()