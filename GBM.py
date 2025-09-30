import numpy as np
import matplotlib.pyplot as plt
# s(t+dt) = st * exp((mu -0.5 * sigma) dt + sigma * np.sqrt(dt) * z)  
class GBM:
  def __init__(self, S0 = 100, mu = 0.05, sigma = 0.2, T = 20, n_steps = 1000, n_paths = 3, seed = 42):
    self.S0 = S0
    self.mu = mu
    self.sigma = 0.2
    self.Time = T
    self.n_steps = n_steps
    self.n_paths = n_paths
    self.seed = 42
    self.t = None
    self.S = None
  
  def compute(self):
    dt = self.Time / self.n_steps
    rng = np.random.default_rng(self.seed)
    self.t = np.linspace(0.0, self.Time, self.n_steps + 1)
    Z = rng.standard_normal(size=(self.n_paths, self.n_steps))
    increments = (self.mu - 0.5 * self.sigma) * dt + self.sigma * np.sqrt(dt) * Z
    logS = np.cumsum(increments, axis = 1)
    logS = np.concatenate((np.zeros((self.n_paths, 1)), logS), axis = 1)
    self.S = self.S0 * np.exp(logS)
    self.plot()
  
  def plot(self):
    plt.figure(figsize = (8, 4.5))
    for i in range(self.S.shape[0]):
      plt.plot(self.t, self.S[i])
    plt.title("Geometric Brownian Motion sample paths")
    plt.xlabel("Time (years)")
    plt.ylabel("S(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
example = GBM()
example.compute()