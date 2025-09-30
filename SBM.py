import numpy as np  
import matplotlib.pyplot as plt

T = 1.0 # total time
n = 1000 # number of steps
dt = T / n # step size
 
t = np.linspace(0, T, n + 1)
W = np.zeros(n + 1)
z = np.random.normal(0, 1, n)

for i in range(1, n + 1):
  W[i] = W[i - 1] + np.sqrt(dt) * z[i - 1] # w(t) - w(t - dt) ~ N(0, dt) 
  
plt.plot(t, W)
plt.title("Brownian Motion")
plt.xlabel("time")
plt.ylabel("stock price")
plt.tight_layout()
plt.show()




