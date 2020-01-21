import random
import math
import time
import os
import re
import pandas as pd # For I/O
from itertools import combinations # For preprocess
from pyspark.sql import *
from pyspark import SparkConf, SparkContext


conf = SparkConf().setMaster("local[*]").setAppName("Project").set("spark.yarn.executor.memoryOverhead","4096").set("spark.driver.maxResultSize","16G").set("spark.driver.memory","16G").set("spark.executor.memory","16G").set("spark.executor.heartbeatInterval", "3600s").set("spark.network.timeout","7200s")
        
sc = SparkContext(conf=conf).getOrCreate()
spark = SparkSession(sc) # for I/O

#################################
#           Preprocess          #
#################################
# Number of User
num_user = 0
# Help building list for the E[X], every movie rated by every users
E_list = []
# User rating for every Movie
User_list = {}
# Max for User_list, means number of movie
num_movie = 0 
# Movie list to record all movie
# movie -> idx
movie_list = {}
# for reference movie with index, since the number of distinct movie != max id of movie
# idx -> movie
check_movie_id = [] 
# for Reference Pearson coef
Pearson_fict = {}
# Test true for demo 
Test = False
data_path = 'data.txt'
output_path = 'output.txt'



def Demo(Test):
    global data_path, output_path
    if Test:    
        data_path = 'demo_data.txt'
        output_path = 'demo_output.txt'


# For building the E_list for building E[x]
def Preprocess():
    global num_movie, num_user, comb, movie_list, check_movie_id
    data = pd.read_csv(data_path, sep="\t", header=None, names = ["Person", "Movie", "Rating"])
    num_user = data['Person'].max()      

    check_movie_id = sorted(list(set(data['Movie'].tolist())))
    num_movie = len(check_movie_id)
    
    for value in range(num_movie):
        movie_list.update({check_movie_id[value] :value})


#################################
#      Read_file / Pearson      #
#################################    
# Make for user
def shuffle(data):
    # (user, movie, point) 
    data = data.split('\t')
    data = list(map(float, data))
    # ((user, (movie, point))
    res = [int(data[0]), [[int(data[1]), data[2]]]]      
    return res 


# user,[user give point for each movie....]
def update_User_list(data):
    # for about 9700 movies
    res_list = num_movie*[0]
    for movie_id, point in data:
        idx = int(movie_id)
        res_list[movie_list[idx]] = point
    return tuple(res_list)

        
# for reducing by key to find E[x] or E[x**2]
def shuf(data):
    # (user, movie, point) 
    data = data.split('\t')
    data = list(map(float, data))
    # (movie, (user, point))
    res = [int(data[1]), [[int(data[0]), data[2]]]]      
    return res 

# movie,[user give point....]
def update_Elist(data):
    # for 610 users
    res_list = num_user*[0]
    for user, point in data:
        idx = int(user)
        res_list[idx-1] = point
    return res_list


def make_Exp(data):
    exp = sum(data)/num_user
    exp2 = sum(map(lambda i : i * i, data))/num_user 
    sqrtexp = math.sqrt(exp2-exp*exp) 
    return [exp, sqrtexp, data]


def Pearson(data):
    key1 = data[0][0]
    key2 = data[0][1]
    
    # find Pearson Coef off key1, key2
    if key1 != key2:
        x_list = data[1][2]
        y_list = data[2][2]
        Exy = sum([a*b for a,b in zip(x_list,y_list)])/num_user
        Pcoef = (Exy-data[1][0]*data[2][0])/(data[1][1]*data[2][1])
        return (key1, [[key2, min(max(Pcoef, 0), 1)]]) 
    return (key1, [[key2, 1]]) 


def sortkey(data):
    sort_list = sorted(data, key = lambda x : x[0])
    result = []
    for key, value in sort_list:
        result.append(value)
    return result
    
            
def Get_Pearson_coef():
    # read Data    
    global User_list, Pearson_list, Pearson_dict
    data = sc.textFile(data_path)
     
    # Make it into (user, (movie rating.......))
    User = data.map(shuffle).reduceByKey(lambda a, b: a + b).mapValues(update_User_list).cache()

    
    # Make them into (Movie, Rating) key-value pair in each data, 
    # Use E_list to build whole list for it
    # Rate for (Movie, (E[x], sqrt(E[x**2]-E[x]**2), (user1_rate, ......))
    Rate = data.map(shuf).reduceByKey(lambda a, b: a + b).mapValues(update_Elist).mapValues(make_Exp)
    #print(Rate.first())
    # make it to ((movie, movie), ((E[x], sqrt(E[x**2]-E[x]**2), (user1_rate, ......),(E[x], sqrt(E[x**2]-E[x]**2), (user1_rate, ......)))
    Rate_comb = Rate.cartesian(Rate).map(lambda x: (tuple((x[0][0], x[1][0])), x[0][1], x[1][1]))
    
    
    # Find Pearson list (movie, Pearson coef movie to all)
    Pearson_list = Rate_comb.map(Pearson).reduceByKey(lambda a, b : a + b).mapValues(sortkey).cache()
    Pearson_list.unpersist()
    
   
    Pearson_dict = Pearson_list.collectAsMap()
    del Pearson_list, Rate, Rate_comb
      
    return User



#################################
#        Rate for others        #
#################################
def Recommendation(data):
    
    user_id = data[0]
    # recording the rating in 9700 movies of a user
    m_list = data[1]
    rate = []
    id_list = []
    
    # total movie rating -> about 9700 movies
    for movie in range(len(m_list)):
        Point = m_list[movie]
        
        # if this movie is not rating
        if Point == 0:
            # find the right list of movie rating
            sub_p = Pearson_dict[check_movie_id[movie]]
            r = sum([m * p for m, p in zip(m_list, sub_p)])
            sum_sub_p = sum(sub_p)
            
            if sum_sub_p != 0:
                r /= sum_sub_p
            
            rate.append(r)
            id_list.append(check_movie_id[movie])
    
    rec_movie_id = id_list[rate.index(max(rate))]
    
    return (user_id, rec_movie_id)


# Find the best recommendation for user 
def Rate(User):
    # Make it to ((User, Rating), P_list), calculate Pearson coef real-time
    Rec = User.map(Recommendation)
    User.unpersist()
    del User
    Result = Rec.collect()
    
    return sorted(Result, key = lambda x : x[0])


def write_file(Recommend):
    with open(output_path, 'w') as f:
        # cloumns means article number not index
        for i in range(len(Recommend)):
            User = Recommend[i][0]
            movie = Recommend[i][1]
            #Rate = Recommend[i][2]
            msg = 'Recommend User {} : Movie {}'.format(User, movie)
            f.writelines(msg+"\n")
    
#################################
#        Excecution Flow        #
#################################
start = time.time()

# Test true for demo 
Test = False

Demo(Test)

print('Preprocessing ...')
Preprocess()

print('prepare User info and Pearson coefficient ...')
User = Get_Pearson_coef()

print('find recommendation ...')
Recommend = Rate(User)

print('writing file...')
write_file(Recommend)
sc.stop()
end = time.time()
print('spend : {}s'.format(end-start))




