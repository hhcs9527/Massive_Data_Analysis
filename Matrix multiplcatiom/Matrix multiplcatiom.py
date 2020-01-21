from pyspark import SparkConf, SparkContext
from operator import itemgetter
conf = SparkConf().setMaster("local[8]").setAppName("First_App")
sc = SparkContext(conf=conf).getOrCreate()

# P = M*N 兩矩陣相乘以 Map Reduce實作

# 讀 inputdata 用來做矩陣相乘, 並分成 M, N 兩組資料
def readfile():
    f = open("500input.txt", "r")
    data = f.readlines()
    f.close()
    M = []
    N = []
    for i, data in enumerate(data):
        data = data.split(',')
        if data[0] == 'M':
            M.append((int(data[1]), int(data[2]), int(data[3])))
        else:
            N.append((int(data[1]), int(data[2]), int(data[3])))
    M = sc.parallelize(M)
    N = sc.parallelize(N)
    return M, N

# 因為 Pij = sum( mik*nkj ), 為了之後join的方便, 
# mik->(k,(i,value)) nkj->(k,(j,value))
def mapper():
    M, N = readfile() 
    M = M.map(lambda entry: (entry[1], (entry[0], entry[2])))
    N = N.map(lambda entry: (entry[0], (entry[1], entry[2])))
    return M, N

# Join可以使得所有 k 為中間值者放在同一個 tuple 中
# 做完 join 就 map 成 ((i,j), mvalue*nvalue)
# 再用 group by 以(i,j)為基礎來分群
def shuffler():    
    M, N = mapper() 
    MN = M.join(N)
    MN_shuffle =(MN.map(lambda entry: ((entry[1][0][0], entry[1][1][0]) ,(entry[1][0][1]*entry[1][1][1]))).groupByKey())
    return MN_shuffle

# 合併最後所有屬於同一個key的值,即完成了 P = M*N
def reducer():
    MN_shuffle = shuffler()
    result = MN_shuffle.map(lambda x: (x[0][0], x[0][1], sum(x[1])))
    return result.collect()


# 寫入檔案，寫入前會先排序，因為相乘完collect()的結果並非要求的順序
# 所以利用 sort, 並以 row 值維修先考量, 再來才是 column 值
def filewriter():
    data = reducer()
    with open ('Outputfile.txt', 'w') as f:
        data = sorted(data, key = itemgetter(0,1))
        for i,towrite in enumerate(data):
            f.write(str(towrite)+'\n')

filewriter()
sc.stop()

