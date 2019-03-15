# python code 문법 테스트 전용 파일

import numpy as np

#N = 10

#b11 = np.array([[1,1],[1,1]])
#b12 = np.array([[2,2],[2,2]])
#b21 = np.array([[3,4],[5,6]])
#b22 = np.array([[1,2],[3,4]])

#b11 = np.vstack([b11 for x in range(N)])
#b12 = np.vstack([b12 for x in range(N)])
#b21 = np.vstack([b21 for x in range(N)])
#b22 = np.vstack([b22 for x in range(N)])

#btot = np.eye(2)

#btot = btot*np.array([[b11,b12],[b21,b22]])

#print(Btot)

#x = np.array([1,2,3,4,5])
#index = list(i for i in range(len(x)) if x[i] > 2)[0]

#print("index = ",index)
# print("value = ",x[index])


def fa():
    b = 1
    c = 1
    return (b,c)

b, c = fa()

print("b = ", b)
print("c = ", c)
