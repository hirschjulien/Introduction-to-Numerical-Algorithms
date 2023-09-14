import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.precision", 15)

#Question B2
x = [2,4,6,8,10,12]
y = [2.31,2.01,1.80,1.66,1.55,1.47]

A = np.array([np.ones(6), x, list(map(lambda x:pow(x,2),x))]).T
c = np.linalg.solve(A.T@A,A.T@y)
print(c)

#Question 2.2
np.set_printoptions(suppress=True, precision=4)
x = [0,1,2,3,4]
y = [4.00,1.74,0.19,-0.54,-0.20]

A = np.array([np.cos(x),list(map(lambda x:pow(np.e,(-x)),x))]).T
c = np.linalg.solve(A.T@A,A.T@y)
print(c)

#Question 2.3
Fx = np.array(list(map(lambda x:c[0]*np.cos(x)+c[1]*pow(np.e,(-x)),x)))
Error = sum(y-Fx)
print(Error)

#Question 3.1
f = lambda x: (x-2)**2*(x-8) 
df = lambda x: 2*(x-2)*(x-8)+(x-2)**2

def Newton(f,df,x0,nmax):
    x = x0
    X = [x]
    for n in range(nmax):
        x = x-f(x)/df(x)
        X.append(x)
    return(X)        

X = np.array(Newton(f,df,4.6,12))
r=2
E = np.abs(r-X)
ratio = np.hstack([0,E[1:]/E[:-1]**2])
table = pd.DataFrame(data={'xn': X, 'error': E, 'ratio': ratio})
print(table)

#Question 3.3
m = 2
def Newton_new(f,df,x0,nmax):
    x = x0
    X = [x]
    for n in range(nmax):
        x = x-m*f(x)/df(x)
        X.append(x)
    return(X)        

X = np.array(Newton_new(f,df,4.6,5))

r=2
E = np.abs(r-X)
ratio = np.hstack([0,E[1:]/E[:-1]**2])
table = pd.DataFrame(data={'xn': X, 'error': E, 'ratio': ratio})
print(table)