import numpy as np
import random

class CEM():
  def __init__(self, func, d, v_min=None, v_max=None, maxits1=500, maxits2=500, N=100, Ne=10, argmin=True):
    self.func = func            # target function
    self.d = d                  # dimension of function input X
    self.maxits1 = maxits1      # maximum iteration of update all dimension
    self.maxits2 = maxits2      # maximum iteration of update one dimension
    self.N = N                  # sample N examples each iteration
    self.Ne = Ne                # using better Ne examples to update mu and sigma
    self.reverse = not argmin   # try to maximum or minimum the target function
    self.init_coef = 1          # sigma initial value
    self.v_min = v_min
    self.v_max = v_max

  def eval(self, instr):#, *args): # if target function has multiple inputs, could specify the inputs and left the last one
    """evalution and return the solution"""
    # initial parameters
    t, mu, sigma = self.__initParams()
#    v = np.random.uniform(size=self.d)

    # random sample all dimension each time
    while t < self.maxits1:
      # sample N data and sort
      x = self.__sampleData(mu, sigma)
      s = self.__functionReward(instr, x)
      s = self.__sortSample(s)
      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
      mu, sigma = self.__updateParams(x, mu, sigma, s)
#      v = x[0]
      t += 1

#    t, mu, sigma = self.__initParams()
    # random sample one dimension each time
#    while t < self.maxits2:
      # sample N data and sort
#      x = self.__sampleData1d(mu, sigma, v)
#      s = self.__functionReward(instr, x)
#      s = self.__sortSample(s)
#      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
#      mu, sigma = self.__updateParams(x, mu, sigma, s)
#      v = x[0]
#      t += 1

    return mu

  def __initParams(self):
    """initial parameters t, mu, sigma"""
    t = 0
    mu = np.zeros(self.d)
    sigma = np.ones(self.d) * self.init_coef
    return t, mu, sigma

  def __updateParams(self, x, mu, sigma, s):
    """update parameters mu, sigma"""
    mu = x[0:self.Ne,:].mean(axis=0)
    sigma = x[0:self.Ne,:].std(axis=0)
    return mu, sigma
    
  def __sampleData(self, mu, sigma):
    """sample N examples"""
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(self.N,))
      if self.v_min is not None and self.v_max is not None:
        sample_matrix[:,j] = np.clip(sample_matrix[:,j], self.v_min[j], self.v_max[j])
    return sample_matrix

  def __sampleData1d(self, mu, sigma, v):
    """sample N examples but only change one dimentional of v"""
    tmp = np.copy(v).reshape((1,-1))
    sample_matrix = np.repeat(tmp, self.N, axis=0)
    j = random.randint(0, self.d-1)
    sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(self.N,))
    if self.v_min is not None and self.v_max is not None:
      sample_matrix[:,j] = np.clip(sample_matrix[:,j], self.v_min[j], self.v_max[j])
    return sample_matrix

  def __functionReward(self, instr, x):
    bi = np.reshape(instr, [1, -1])
    bi = np.repeat(bi, self.N, axis=0)
    return zip(x, self.func(bi, x))

#  def __functionReward(self, args, x):
#    """get function return"""
#    s = []
#    for i in range(self.N):
#      s.append((x[i], self.func(*args, x[i])))
#    return s

  def __sortSample(self, s):
    """sort data by function return"""
    s = sorted(s, key=lambda x: x[1], reverse=self.reverse)
    return s
    
def func(a1, a2):
  c = a1 - a2
  return c[0]*c[0] + c[1]*c[1]

if __name__ == '__main__':
  cem = CEM(func, 2)
  t = np.array([1,2])
  v = cem.eval(t)
  print(v, func(t, v))
