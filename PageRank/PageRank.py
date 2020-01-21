import time
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[8]").setAppName("hw2")
sc = SparkContext(conf=conf).getOrCreate()

class PageRank():
    def __init__(self, times, beta):
        self.times = times
        self.readfile()
        self.beta = beta
        
    # read file and filter the # which in the reading data
    def readfile(self):
        data  = sc.textFile('p2p-Gnutella04.txt')
        self.data = data.filter(lambda x: '#' not in x)

        
    # from 1 2 -> (1,2) & Create the init state
    def preprocess(self):
        process = self.data.map(lambda each: (int(each.split('\t')[0]), int(each.split('\t')[1])))
        self.N = max(process.max()[0], process.max(lambda x :x[1])[1]) + 1
        # Create the init state
        init = [(i, 1/self.N) for i in range(self.N)]
        self.state = sc.parallelize(init)
        return process
     
        
    # transform sparse matrix to linked list like data (for multiplication) 
    # 1 2
    # 1 3 
    # to (1,(2, 1/2)), (1,(3, 1/2)) 
    # 表示說node 1 會到 node2 的機率為 1/2 以及 node3的機率 1/2 
    def Mapper(self, process):
        data = process.groupByKey().map(lambda element: ((element[0], 1/len(element[1])), element[1]))
        # Reverse the groupby & to (1,(2, 1/2)), (1,(3, 1/2)) 
        transition = data.flatMapValues(lambda x : x).map(lambda x :  (x[0][0], (x[1],x[0][1])))
        #print(transition.top(5))
        return transition
    
    
    # Do the matrix mutiplication part
    def Reducer(self):   
        self.transition = self.Mapper(self.preprocess()).cache()
            #transition 
        beta = self.beta
        N = self.N
        for i in range(self.times):
            # 先將node相同的join起來，to distribute the probability of node to others throgh transition matrix
            result = self.transition.join(self.state)
            # (1, ((2, 0.5), 0.2) -> represent node 1 have 0.2 point, 
            # there is 50% chance for node 1 to node 2
            # after the next step, tuple will become (2, 0.5*0.2)
            # indicate that in this part, node 2 get 0.5*0.2 point, then reduceByKey.
            # And we can find the point for each node

            self.state = result.map(lambda x : (x[1][0][0], beta * x[1][0][1]*x[1][1])).reduceByKey(lambda x, y: x + y)
            sumofvalue = self.state.values().sum()
            # need to re-insert the leaked pagerank
            self.state = self.state.mapValues(lambda x : x + (1-sumofvalue)/N)
    
    
    def filewriter(self):
        data = self.state.top(10, key = lambda x : x[1])
        with open ('Outputfile.txt', 'w') as f:
            for i,towrite in enumerate(data):
                f.write(str(towrite)+'\n')
                
                
    def forward(self):
        self.Reducer()
        self.filewriter()
        return self.state.top(10, key = lambda x : x[1])
        



tStart = time.time()

PR = PageRank(times = 20, beta = 0.8)
ans = PR.forward()

tEnd = time.time()
print()
print ('Final State : ', ans)
print()
print ('Spend : ',tEnd - tStart)
sc.stop()