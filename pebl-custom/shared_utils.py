"""Miscellaneous utility functions."""


def as_list(c):
    """Ensures that the result is a list.

    If input is a list/tuple/set, return it.
    If it's None, return empty list.
    Else, return a list with input as the only element.
    
    """

    if isinstance(c, (list,tuple,set)):
        return c
    elif c is None:
        return []
    else:
        return [c]


def unzip(l, *jj):
    """Opposite of zip().

    jj is a tuple of list indexes (or keys) to extract or unzip. If not
    specified, all items are unzipped.

    """
	
    if jj==():
	    jj=range(len(l[0]))
    rl = [[li[j] for li in l] for j in jj] # a list of lists
    if len(rl)==1:
        rl=rl[0] #convert list of 1 list to a list
    return rl

def extended_property(func):
  """Function decorator for defining property attributes

  The decorated function is expected to return a dictionary
  containing one or more of the following pairs:

      * fget - function for getting attribute value
      * fset - function for setting attribute value
      * fdel - function for deleting attribute

  """
  return property(doc=func.__doc__, **func())

