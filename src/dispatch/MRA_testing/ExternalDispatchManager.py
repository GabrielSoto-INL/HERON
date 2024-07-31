
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Class for managing interactions with the Dispatchers.
"""


import os
import sys
import pickle as pk
from time import time as run_clock

import numpy as np
from typing_extensions import final

# make functions findable
cwd = os.getcwd()

sys.path.append(cwd)
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir, os.pardir)))
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir)))

from src import _utils as hutils
from src import SerializationManager

try:
  from ravenframework.PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
  import TEAL.src as TEAL
except ModuleNotFoundError:
  raven_path = hutils.get_raven_loc()
  sys.path.append(raven_path)
  from ravenframework.PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
  sys.path.pop()

  cashflow_path = os.path.abspath(os.path.join(hutils.get_cashflow_loc(raven_path=raven_path), '..'))
  sys.path.append(cashflow_path)
  import TEAL.src as TEAL

from src.DispatchManager import DispatchRunner

class ExternalDispatchRunner(DispatchRunner):
  """
    Manages the interface between RAVEN and running the dispatch
  """
  # TODO move naming templates to a common place for consistency!
  naming_template = {
    'comp capacity': '{comp}_capacity',
    'dispatch var': 'Dispatch__{comp}__{tracker}__{res}',
    'cashflow alpha': '{comp}_{cf}_alpha',
    'cashflow driver': '{comp}_{cf}_driver',
    'cashflow reference': '{comp}_{cf}_reference',
    'cashflow scale': '{comp}_{cf}_scale',
  }

  #####################
  # API
  def extract_variables_no_raven(self, raven_dict):
    """
      Extract variables from RAVEN and apply them to HERON objects
      @ In, raven_dict, dict, RAVEN input dictionary
      @ Out, pass_vars, dict, variables to pass to dispatcher
    """
    pass_vars = {}
    history_structure = {}
    # investigate sources for required ARMA/CSV information
    for source in self._sources:
      if source.is_type('ARMA') or source.is_type("CSV"):
        # get structure of ARMA/CSV
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = raven_dict.get(v, None)

    # get the key to mapping RAVEN multidimensional variables
    if raven_dict.get('_indexMap'):
      pass_vars['_indexMap'] = raven_dict.get('_indexMap')[0] # 0 is only because of how RAVEN EnsembleModel handles variables
      # collect all indices # TODO limit to those needed by sources?
      for target, required_indices in pass_vars['_indexMap'].items():
        for index in filter(lambda idx: idx not in pass_vars, required_indices):
          pass_vars[index] = raven_dict.get(index)
    else:
      # NOTE this should ONLY BE POSSIBLE if no ARMAs or CSVs are in use!
      pass

    # variable for "time" discretization, if present
    year_var = self._case.get_year_name()
    time_var = self._case.get_time_name()
    time_vals = raven_dict.get(time_var, None)
    if time_vals is not None:
      pass_vars[time_var] = time_vals

    # TODO references to all ValuedParams should probably be registered somewhere
    # like maybe in the VPFactory, then we can loop through and look for info
    # that we know from Outer and fill in the blanks? Maybe?
    for magic in self._case.dispatch_vars.keys():
      val = raven_dict.get(f'{magic}_dispatch', None)
      if val is not None:
        pass_vars[magic] = float(val)

    # component capacities
    for comp in self._components:
      name = self.naming_template['comp capacity'].format(comp=comp.name)
      update_capacity = raven_dict.get(name) # TODO is this ever not provided?
      if update_capacity is not None:
        comp.set_capacity(update_capacity)
        pass_vars[f'{comp.name}_capacity'] = update_capacity

      # component cashflows
      # TODO this should be more automated - registry?
      for cf in comp.get_cashflows():
        for att in ['alpha', 'driver', 'reference', 'scale']:
          cf_att = self.naming_template[f'cashflow {att}'].format(comp=comp.name, cf=cf.name)
          update_cf_att = raven_dict.get(cf_att) # TODO
          if update_cf_att is not None:
            pass_vars[cf_att] = update_cf_att

    # TODO other case, component properties

    # check macro parameter
    if year_var in dir(raven_dict):
      year_vals = raven_dict.get(year_var)
      year_size = year_vals.size
      project_life = hutils.get_project_lifetime(self._case, self._components) - 1 # 1 for construction year
      if year_size != project_life:
        raise RuntimeError(f'Provided macro variable "{year_var}" is length {year_size}, ' +
                           f'but expected project life is {project_life}! ' +
                           f'"{year_var}" values: {year_vals}')

    # load ARMA signals
    for source in self._sources:
      if source.is_type('ARMA'):
        vars_needed = source.get_variable()
        for v in vars_needed:
          vals = raven_dict.get(v, None)
          # checks
          if vals is None:
            raise RuntimeError(f'HERON: Expected ARMA variable "{v}" was not passed to DispatchManager!')
          pass_vars[v] = vals
    return pass_vars


  def _build_econ_objects(self, heron_case, heron_components, project_life):
    """
      Generates CashFlow.CashFlow instances from HERON CashFlow instances
      Note the only reason there's a difference is because HERON needs to retain some level of
      flexibility in the parameter values until this method is called, whereas CashFlow expects
      them to be evaluated.
      @ In, heron_case, HERON Case instance, global HERON settings for this analysis
      @ In, heron_components, list, HERON component instances
      @ In, project_life, int, number of years to evaluate project
      @ Out, global_settings, CashFlow.GlobalSettings instance, settings for CashFlow analysis
      @ Out, teal_components, dict, CashFlow component instances
    """
    heron_econs = list(comp.get_economics() for comp in heron_components)
    # build global econ settings for CashFlow
    global_params = heron_case.get_econ(heron_econs)
    global_settings = TEAL.CashFlows.GlobalSettings()
    global_settings.setParams(global_params)
    global_settings.setVerbosity(global_params.get('verbosity',0)) # NOTE: this is verbosity in economics
    # build TEAL CashFlow component instances
    teal_components = {}
    for c, cfg in enumerate(heron_econs):
      # cfg is the cashflowgroup connected to the heron component
      # get the associated heron component
      heron_comp = heron_components[c]
      comp_name = heron_comp.name
      # build TEAL equivalent component
      teal_comp = TEAL.CashFlows.Component()
      teal_comp_params = {'name': comp_name,
                        'Life_time': cfg.get_lifetime(),
                        # TODO StartTime, Repetitions, custom tax/inflation rate
                       }
      teal_comp.setParams(teal_comp_params)
      teal_components[comp_name] = teal_comp
      # create all the TEAL.CashFlows (teal_cf) for the TEAL.Component
      teal_cfs = []
      for heron_cf in cfg.get_cashflows():
        cf_name = heron_cf.name
        cf_type = heron_cf.get_type()
        cf_taxable = heron_cf.is_taxable()
        cf_inflation = heron_cf.is_inflation()
        cf_mult_target = heron_cf.is_mult_target()

        # skip NPV-exempt cashflows
        # This means that the user has specified this cashflow should not be included in the NPV calculation.
        if heron_cf.is_npv_exempt():
          continue
        # the way to build it slightly changes depending on the CashFlow type
        if cf_type == 'repeating':
          teal_cf = TEAL.CashFlows.Recurring()
          # NOTE: the params are listed in order of how they're read in TEAL.CashFlows.CashFlow.setParams
          teal_cf_params = {'name': cf_name,
                          # driver: comes later
                          'tax': cf_taxable,
                          'inflation': cf_inflation,
                          'mult_target': cf_mult_target,
                          # multiply: do we ever use this?
                          # alpha: comes later
                          # reference: not relevant for recurring
                          'X': 1.0,
                          # depreciate: not relevant for recurring
                          }
          teal_cf.setParams(teal_cf_params)
          teal_cf.initParams(project_life)
        elif cf_type == 'one-time':
          teal_cf = TEAL.CashFlows.Capex()
          teal_cf.name = cf_name
          teal_cf_params = {'name': cf_name,
                            'driver': 1.0, # handled in segment_cashflow
                            'tax': cf_taxable,
                            'inflation': cf_inflation,
                            'mult_target': cf_mult_target,
                            # multiply: do we ever use this?
                            'alpha': 1.0, # handled in segment_cashflow
                            'reference': 1.0, # actually handled in segment_cashflow
                            'X': 1.0,
                            # depreciate: handled in segment_cashflow
                           }
          teal_cf.setParams(teal_cf_params)
          teal_cf.initParams(teal_comp.getLifetime())
          # alpha, driver aren't known yet, so set those later
        else:
          raise NotImplementedError(f'Unknown HERON CashFlow Type: {cf_type}')
        # store new object
        teal_cfs.append(teal_cf)
      teal_comp.addCashflows(teal_cfs)
    return global_settings, teal_components


  def _final_cashflow(self, meta, final_components, final_settings) -> dict:
    """
      Perform final cashflow calculations using TEAL.
      @ In, meta, dict, auxiliary information
      @ In, final_components, list, completed TEAL component objects
      @ In, final_settings, TEAL.Settings, completed TEAL settings object
      @ Out, cf_metrics, dict, values for calculated metrics
    """
    print('****************************************')
    print('* Starting final cashflow calculations *')
    print('****************************************')
    raven_vars = meta['HERON']['RAVEN_vars_full']
    # DEBUGG
    print('DEBUGG CASHFLOWS')
    for comp_name, comp in final_components.items():
      print(f' ... comp {comp_name} ...')
      for cf in comp.getCashflows():
        print(f' ... ... cf {cf.name} ...')
        print(f' ... ... ... D: {cf.getDriver()}')
        print(f' ... ... ... a: {cf.getAlpha()}')
        print(f' ... ... ... Dp: {cf.getReference()}')
        print(f' ... ... ... x: {cf.getScale()}')
        if hasattr(cf, '_yearlyCashflow'):
          print(f' ... ... ... hourly: {cf.getYearlyCashflow()}')
    # END DEBUGG
    cf_metrics = TEAL.main.run(final_settings, list(final_components.values()), raven_vars)
    # DEBUGG
    print('****************************************')
    print('DEBUGG final cashflow metrics:')
    for k, v in cf_metrics.items():
      if k not in ['outputType', 'all_data']:
        print('  ', k, v)
    print('****************************************')
    # END DEBUGG
    return cf_metrics

# class DispatchManager(ExternalModelPluginBase):
#   """
#     A plugin to run heron.lib
#   """

#   def initialize(self, container, runInfoDict, inputFiles):
#     """
#       Method to initialize the DispatchManager plugin.
#       @ In, container, object, external 'self'
#       @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
#       @ In, inputFiles, list, not used
#       @ Out, None
#     """
#     pass

#   def _readMoreXML(self, raven, xml):
#     """
#       Reads additional inputs for DispatchManager
#       @ In, raven, object, variable-storing object
#     """
#     respec = xml.find('respecTime')
#     if respec is not None:
#       try:
#         stats = [int(x) for x in respec.text.split(',')]
#         raven._override_time = stats
#         np.linspace(*stats) # test it out
#       except Exception:
#         raise IOError('DispatchManager xml: respec values should be arguments for np.linspace! Got', respec.text)

#   def run(self, raven, raven_dict):
#     """
#       # TODO split into dispatch manager class and dispatch runner external model
#       API for external models.
#       This is run as part of the INNER ensemble model, run after the synthetic history generation
#       @ In, raven, object, RAVEN variables object
#       @ In, raven_dict, dict, additional RAVEN information
#       @ Out, None
#     """
#     path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
#     # build runner
#     runner = DispatchRunner()
#     # load library file
#     runner.load_heron_lib(path)
#     # load data from RAVEN
#     raven_vars = runner.extract_variables(raven, raven_dict)
#     # TODO clustering, multiyear, etc?
#     # add settings from readMoreXML
#     override_time = getattr(raven, '_override_time', None)
#     if override_time is not None:
#       runner.override_time(override_time) # TODO setter
#     dispatch, metrics, tot_activity = runner.run(raven_vars)
#     runner.save_variables(raven, dispatch, metrics, tot_activity)


