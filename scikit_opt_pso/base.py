import types
import warnings

from abc import ABCMeta     # Abstract Base Classes

class SkoBase(metaclass=ABCMeta):
  def register(self, operator_name, operator, *args, **kwargs):
    '''
    register udf to the class
    :param operator_name: string
    :param operator: a function, operator itself
    :param args: args of operator
    :param kwargs: kwargs of operator
    :return:
    '''

    def operator_wrapper(*wrapper_args):
      return operator(*(wrapper_args + args), **kwargs)

    setattr(self, operator_name, types.MethodType(operator_wrapper, self))
    return self

  def fit(self, *args, **kwargs):
    warnings.warn('.fit() will be deprecated in the future. use .run() instead.',
                    DeprecationWarning)
    return self.run(*args, **kwargs)