import operator

import matplotlib
import matplotlib.pyplot as plt

from numpy import *

'''
plt.plot([1,2,3])  
plt.ylabel('some numbers')  
plt.show()  
'''


lt = [1, 2, 3, 4, 5, 6]

def add(num):

    return num + 1

 

rs = mat(map(add, lt))

print (rs) #[2,3,4,5,6,7]


rand = random.rand(1,2)
print(rand)
rand1 = 1.0 +rand
print(rand1)

data_mata = {"data": {20: {20141023: 3.6, 20141024: 3.6}},
             "feed_data": {20141023: [1, 2, 3], 20141024: [1, 2, 3]}},

if __name__ == '__main__':
    with open('data_mata.txt', 'wb') as f:
        f.write(data_mata)