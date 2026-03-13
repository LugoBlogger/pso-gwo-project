"""
Logs
- [2025/11/27]     
  Numba version of psoalgo. for rng.uniform(low, high, size), numba
  does not support low and high

"""

import numpy as np

from numba import njit

@njit
def run_pso_numba(obj_func, N_particles, o_arr, D, M_tensor, bounds, params, 
                  rng, iter_max=100):

  w, phi_p, phi_g = params
  # x_arr = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_particles, D))
  x_arr = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*rng.random((N_particles, D))
  f_x_arr = obj_func(x_arr, o_arr, D, M_tensor).T

  p_arr = x_arr.copy()
  f_p_arr = f_x_arr.copy()
  
  g_best = p_arr[np.argmin(f_p_arr)][np.newaxis, :]
  f_g_best = obj_func(g_best, o_arr, D, M_tensor)

  bounds_range = (bounds[:, 1] - bounds[:, 0]).flatten()
  # v_arr = rng.uniform(-bounds_range, bounds_range, (N_particles, D))
  v_arr = -bounds_range[np.newaxis, :] + 2*bounds_range[np.newaxis, :]*rng.random((N_particles, D))

  history_p_arr = np.zeros((iter_max+2, *p_arr.shape))
  history_p_arr[0] = p_arr.copy()

  history_f_p_arr = np.zeros((iter_max+2, *f_p_arr.shape))
  history_f_p_arr[0] = f_p_arr.copy()

  history_g_best = np.zeros((iter_max+2, *g_best.shape))
  history_g_best[0] = g_best[0].copy()
  
  history_f_g_best = np.zeros((iter_max+2, *f_g_best.shape))
  history_f_g_best[0] = f_g_best[0].copy()

  run_algo = True
  iter_start = 0
  while run_algo:
    r_p = rng.uniform(size=(N_particles, D))
    r_g = rng.uniform(size=(N_particles, D))

    # -- update the particle's velocity
    v_arr = w*v_arr + phi_p*r_p*(p_arr - x_arr) + phi_g*r_g*(g_best - x_arr)

    # -- update the particle's velocity
    x_arr += v_arr

    # -- if the particle's position is outside the domain, we set it
    # to the boundary points. This is the constraint of our optimization
    x_arr = np.clip(x_arr, bounds[:, 0], bounds[:, 1])

    f_x_arr = obj_func(x_arr, o_arr, D, M_tensor).T
    # f_p_i = obj_func(p_arr, o_arr, D, M_tensor).T

    # # -- update particle's best known position
    mask = f_x_arr < f_p_arr
    # print(f"mask.shape", mask.shape)
    p_arr = np.where(mask, x_arr, p_arr)
    f_p_arr = np.where(mask, f_x_arr, f_p_arr)

    # # -- update the swarm's best known position
    g_best = p_arr[np.argmin(f_p_arr)][np.newaxis, :]
    f_g_best = obj_func(g_best, o_arr, D, M_tensor)

    iter_start += 1

    history_p_arr[iter_start] = p_arr.copy()
    history_f_p_arr[iter_start] = f_p_arr.copy()
    history_g_best[iter_start] = g_best[0].copy()
    history_f_g_best[iter_start] = f_g_best[0].copy()

    if iter_start > iter_max:
      run_algo = False

  return history_p_arr, history_f_p_arr, history_g_best, history_f_g_best
