# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""
import numpy as np

synthetic_years_to_sample = [2022, 2032]
number_of_years_per_set   = [10, 10]

def load_time_set_parameters(projectTime):
  """
    Load time parameters set above by the user
    @ In, projectTime, int, project life time
    @ Out, time_set_params, dict, time set parameters set by user
  """
  assert np.sum(number_of_years_per_set) == projectTime
  time_set_params = {}
  time_set_params['years'] = synthetic_years_to_sample
  time_set_params['n_years_per_set'] = number_of_years_per_set
  return time_set_params
