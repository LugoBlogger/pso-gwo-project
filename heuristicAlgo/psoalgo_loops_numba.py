"""
Logs
- [2025/12/01]    
  Wrapping PSO algo with numba without vectorization
"""

import numpy as np

from numba import njit

@njit
def run_pso_loops_numba(obj_func, N_particles, o_arr, D, M_tensor, bounds, 
                        params, rng, iter_max=100):
  w, phi_p, phi_g = params

  x_arr = np.zeros((N_particles, D))
  p_arr = np.zeros((N_particles, D))
  f_x_arr = np.zeros(N_particles)
  f_p_arr = np.zeros(N_particles)
  for idx in range(N_particles):
    for dim in range(D):
      x_arr[idx, dim] = bounds[dim, 0] + (bounds[dim, 1] - bounds[dim, 0])*rng.random()
      p_arr[idx, dim] = x_arr[idx, dim]

  
    f_x_arr[idx] = obj_func(p_arr[idx], o_arr, D, M_tensor)
    f_p_arr[idx] = f_x_arr[idx]

  g_best = p_arr[np.argmin(f_p_arr)]
  f_g_best = obj_func(g_best, o_arr, D, M_tensor)

  bounds_range = np.zeros(D)
  for dim in range(D):
    bounds_range[dim] = bounds[dim, 1] - bounds[dim, 0]

  # v_arr = rng.uniform(-bounds_range, bounds_range, (N_particles, D))
  v_arr = np.zeros((N_particles, D))
  for idx in range(N_particles):
    for dim in range(D):
      v_arr[idx, dim] = -bounds_range[dim] + 2*bounds_range[dim]*rng.random()

  history_p_arr = np.zeros((iter_max+2, *p_arr.shape))
  history_f_p_arr = np.zeros((iter_max+2, *f_p_arr.shape))
  history_g_best = np.zeros((iter_max+2, *g_best.shape))
  history_f_g_best = np.zeros((iter_max+2))
  for idx in range(N_particles):
    for dim in range(D):
      history_p_arr[0, idx, dim] = p_arr[idx, dim]

    history_f_p_arr[0, idx] = f_p_arr[idx]
    
  for dim in range(D):
    history_g_best[0, dim] = g_best[dim]
  history_f_g_best[0] = f_g_best

  run_algo = True
  iter_start = 0


  while run_algo:
    r_p = rng.random(size=(N_particles, D))  
    r_g = rng.random(size=(N_particles, D))
    
    # -- update the particle's velocity
    for idx in range(N_particles):
      for dim in range(D):
        v_arr[idx, dim] = w*v_arr[idx, dim] \
          + phi_p*r_p[idx, dim]*(p_arr[idx, dim] - x_arr[idx, dim]) \
          + phi_g*r_g[idx, dim]*(g_best[dim] - x_arr[idx, dim])


    # -- update the particle's position
    for idx in range(N_particles):
      for dim in range(D):
        x_arr[idx, dim] += v_arr[idx, dim]

        # -- if the particle's position is outside the domain, we set it
        # to the boundary points. This is the constraint of our optimization
        if x_arr[idx, dim] < bounds[dim, 0]:
          x_arr[idx, dim] = bounds[dim, 0]
        
        if x_arr[idx, dim] > bounds[dim, 1]:
          x_arr[idx, dim] = bounds[dim, 1]

      f_x_arr[idx] = obj_func(x_arr[idx], o_arr, D, M_tensor)
    
    # -- update particle's best known position
    # print(f"mask.shape", mask.shape)
    for idx in range(N_particles):
      if f_x_arr[idx] < f_p_arr[idx]:
        for dim in range(D):
          p_arr[idx, dim] = x_arr[idx, dim]

        f_p_arr[idx] = f_x_arr[idx]

    # -- update the swarm's best known position
    g_best = p_arr[np.argmin(f_p_arr)]
    f_g_best = obj_func(g_best, o_arr, D, M_tensor)

    iter_start += 1

    for idx in range(N_particles):
      for dim in range(D):
        history_p_arr[iter_start, idx, dim] = p_arr[idx, dim]

      history_f_p_arr[iter_start, idx] = f_p_arr[idx]
        
    for dim in range(D):
      history_g_best[iter_start, dim] = g_best[dim]
    
    history_f_g_best[iter_start] = f_g_best

    if iter_start > iter_max:
      run_algo = False

  return history_p_arr, history_f_p_arr, history_g_best, history_f_g_best