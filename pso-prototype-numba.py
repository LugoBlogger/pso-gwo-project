import time
import numpy as np


from heuristicAlgo import psoalgo_loops_numba
from numba import njit

@njit
def obj_func_1(x, o, D, M, a=0.5, b=3, k_max=20):
  f_bias_11 = 90
  z = np.dot((x - o), M)

  k_arr = np.arange(k_max + 1)

  # return a**k_arr 
  term1 = np.sum((a**k_arr)[:, np.newaxis, np.newaxis, np.newaxis]\
    *np.cos((2.*np.pi*b**k_arr)[:, np.newaxis, np.newaxis, np.newaxis] * (z + 0.5)), axis=0)
  term2 = D*np.sum(a**k_arr * np.cos(2.*np.pi*b**k_arr * 0.5), axis=0)
  return np.sum(term1, axis=-1) - term2 + f_bias_11

@njit
def obj_func_1_loops(x, o, D, M, a=0.5, b=3, k_max=20):
  f_bias_11 = 90
  z = np.zeros(D)
  row_M = 2
  for dim in range(D):
    for j in range(row_M):
      z[dim] += (x[j] - o[j]) * M[j, dim]

  k_arr = np.arange(k_max + 1)

  term1 = 0
  for dim in range(D):
    term1sub = 0
    for k in range(k_max):
      term1sub += a**k_arr[k] * np.cos(2.*np.pi*b**k_arr[k]*(z[dim] - 0.5))

    term1 += term1sub
  
  term2 = 0
  for k in range(k_max):
    term2 += a**k_arr[k] * np.cos(2.*np.pi*b**k_arr[k] * 0.5)

  return term1 - D*term2 + f_bias_11

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


if __name__ == "__main__":
  
  M_D2, o_arr = get_M_D2_and_o_arr()

  D = 2
  bounds = np.reshape([-0.5, 0.5]*D, (-1, 2))     # 2 in here is low and high values

  # N_particles = 50
  # N_particles = 400 
  # N_particles = 1000
  N_particles = 5000

  # o_sol = o_arr[:D][np.newaxis, :]
  o_sol = o_arr[:D]
  # M_tensor = M_D2[np.newaxis, :, :]
  M_tensor = M_D2.copy()

  seed = 25_12_01
  rng = np.random.default_rng(seed)
  # rng = np.random.default_rng()
  
  iter_max = 100

  runs_history = {
    "best_solution": [], 
    "best_fitness": [], 
    "history_p_arr": [],
    "history_f_p_arr": []
  }

  # num_of_runs = 2
  # num_of_runs = 10      # xx secs / 400 particles
  # num_of_runs = 100     # xx secs / 400 particles
                          # xx secs / 50 particles

  num_of_runs = 1000     # 16 - 23 secs / 50 particles
                         # 98 - 149 secs / 400 particles
                         # 238 - 374 secs / 1000 particles
                         # 1230 - secs / 5000 particles

  w = 0.5
  phi_p = 0.3
  phi_g = 0.3
  params = [w, phi_p, phi_g]

  # obj_func_1.__dict__["mode"] = "vectorization"

  start_time = time.perf_counter()

  for i in range(num_of_runs):
    print(f"run: {i}")
    # data_history = psoalgo.run_pso_without_store_x_arr(
    #   obj_func_1, coor_particles, o_sol, D, M_tensor, bounds, rng=rng, 
    #   iter_max=iter_max)

    # history_p_arr, history_f_p_arr, history_g_best, history_f_g_best \
    #   = psoalgo_loops_numba.run_pso_loops_numba(
    #       obj_func_1, N_particles, o_sol, D,
    #       M_tensor, bounds, params, rng, iter_max=iter_max)

    history_p_arr, history_f_p_arr, history_g_best, history_f_g_best \
      = psoalgo_loops_numba.run_pso_loops_numba(
          obj_func_1_loops, N_particles, o_sol, D,
          M_tensor, bounds, params, rng, iter_max=iter_max)

    # best_solution = data_history["g_best"]
    # best_fitness = obj_func_1(best_solution, o_sol, D, M_tensor)
    runs_history["best_solution"].append(history_g_best)
    runs_history["best_fitness"].append(history_f_g_best)
    runs_history["history_p_arr"].append(history_p_arr)
    runs_history["history_f_p_arr"].append(history_f_p_arr)

  print(f"Computational time: {time.perf_counter() - start_time:.2f} s")