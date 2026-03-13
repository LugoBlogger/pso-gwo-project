import time

import numpy as np
from scikit_opt_pso.pso_scikit_opt import PSO
from scikit_opt_pso.tools import set_run_mode


def obj_func_1(x, o, D, M, a=0.5, b=3, k_max=20):
  f_bias_11 = 90
  z = np.dot((x - o), M)

  k_arr = np.arange(k_max + 1)

  # return a**k_arr 
  term1 = np.sum((a**k_arr)[:, np.newaxis, np.newaxis, np.newaxis]\
    *np.cos((2.*np.pi*b**k_arr)[:, np.newaxis, np.newaxis, np.newaxis] * (z + 0.5)), axis=0)
  term2 = D*np.sum(a**k_arr * np.cos(2.*np.pi*b**k_arr * 0.5), axis=0)
  return np.sum(term1, axis=-1) - term2 + f_bias_11


def get_M_D2_and_o_arr():
  M_D2 = np.zeros((2, 2)) 
  with open("../cec-2005/weierstrass_M_D2.txt") as fp:
    for i, line in enumerate(fp):
      M_D2[i] = np.array(list(map(float, line.strip().split())))

  o_arr = None
  with open("../cec-2005/weierstrass_data.txt") as fp:
    for i, line in enumerate(fp):
      o_arr = np.array(list(map(float, line.strip().split())))
  
  return M_D2, o_arr


def func(x):
  D = 2
  
  M_D2, o_arr = get_M_D2_and_o_arr()

  o_sol = o_arr[:D][np.newaxis]
  M_tensor = M_D2[np.newaxis, :, :]

  return obj_func_1([x], o_sol, D, M_tensor)


if __name__ == "__main__":
  iter_max = 100
  # iter_max = 2      # for testing

  # N_particles = 50
  # N_particles = 400
  # N_particles = 1000
  N_particles = 5000

  num_of_runs = 1000      #  45 - 68 secs / 50 particles
                          #  106 - 165 secs / 400 particles
                          #  238 - 376 secs / 1000 particles
                          #  1081 - secs / 5000 particles


  pso_runs_obj = []

  start_time = time.perf_counter()

  for i in range(num_of_runs):
    print(f"run: {i}")
    set_run_mode(func, "vectorization")
    pso = PSO(func=func, n_dim=2, pop=N_particles, max_iter=iter_max, 
              lb=[-0.5, -0.5], ub=[0.5,0.5], w=0.5, c1=0.3, c2=0.3)
    pso.run()
    pso_runs_obj.append(pso)
  
  print(f"Computational time: {time.perf_counter() - start_time:.2f} s")