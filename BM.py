import numpy as np
import matplotlib.pyplot as plt
#X(t+Δt)​=X(t)​+ μ * dt + sigma * np.sqrt(dt) * Z
#X(t) = X(0) + μ * t + sigma * W(t) 
class BM:
  def __init__(self, X0 = 0, mu = 0.05, sigma = 0.2, T = 20, n_steps = 1000, n_paths = 3, seed = 42):
    self.X0 = X0
    self.mu = mu
    self.sigma = 0.2
    self.Time = T
    self.n_steps = n_steps
    self.n_paths = n_paths
    self.seed = 42
    self.t = None
    self.X = None
    
  def compute(self):
    dt = self.Time / self.n_steps
    rng = np.random.default_rng(self.seed)
    self.t = np.linspace(0.0, self.Time, self.n_steps + 1)
    Z = rng.standard_normal(size=(self.n_paths, self.n_steps))
    increments = self.mu * dt + self.sigma * np.sqrt(dt) * Z 
    self.X = np.cumsum(increments, axis = 1)
    self.X = np.concatenate((np.full((self.n_paths, 1), self.X0), self.X), axis = 1)
    self.plot()
  def plot(self):
    plt.figure(figsize = (8, 4.5))
    for i in range(self.X.shape[0]):
      plt.plot(self.t, self.X[i])
    plt.title("Geometric Brownian Motion sample paths")
    plt.xlabel("Time (years)")
    plt.ylabel("S(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
example = BM()
example.compute()