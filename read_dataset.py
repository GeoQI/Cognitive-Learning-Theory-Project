"""Classes and functions for working with datasets."""

from __future__ import with_statement
import re
import numpy as N
from shared_utils import as_list
#
# Exceptions
#
class ParsingError(Exception): 
    """Error encountered while parsing an ill-formed datafile."""
    pass

class ClassVariableError(Exception):
    """Error with a class variable."""
    msg = """Data for class variables must include only the labels specified in
    the variable annotation."""


#
# Variables and Samples
#
class Annotation(object):
    """Additional information about a sample or variable."""

    def __init__(self, name, *args):
        # *args is for subclasses
        self.name = str(name)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__,  self.name)

class Sample(Annotation):
    """Additional information about a sample."""
    pass 

class Variable(Annotation): 
    """Additional information about a variable."""
    arity = -1

class ContinuousVariable(Variable): 
    """A variable from a continuous domain."""
    def __init__(self, name, param):
        self.name = str(name)

class DiscreteVariable(Variable):
    """A variable from a discrete domain."""
    def __init__(self, name, param):
        self.name = str(name)
        self.arity = int(param)

class ClassVariable(DiscreteVariable):
    """A labeled, discrete variable."""
    def __init__(self, name, param):
        self.name = str(name)
        self.labels = [l.strip() for l in param.split(',')]
        self.label2int = dict((l,i) for i,l in enumerate(self.labels))
        self.arity = len(self.labels)
        

def maximum_entropy_discretize(indata, includevars=None, excludevars=[], numbins=3):
    """Performs a maximum-entropy discretization of data in-place.
    
    Requirements for this implementation:

        1. Try to make all bins equal sized (maximize the entropy)
        2. If datum x==y in the original dataset, then disc(x)==disc(y) 
           For example, all datapoints with value 3.245 discretize to 1
           even if it violates requirement 1.
     
     Example:

         input:  [3,7,4,4,4,5]
         output: [0,1,0,0,0,1]
        
         Note that all 4s discretize to 0, which makes bin sizes unequal.                                                 

    """

    # includevars can be an atom or list
    includevars = as_list(includevars) 
   
    # determine the variables to discretize
    includevars = includevars or range(indata.variables.size)
    includevars = [v for v in includevars if v not in excludevars]
    
    binsize = indata.samples.size//numbins
    for v in includevars:
        vdata = indata.observations[:,v]
        argsorted = vdata.argsort()
        binedges = [vdata[argsorted[binsize*b - 1]] for b in range(numbins)][1:]
        indata.observations[:,v] = N.searchsorted(binedges, vdata)

        oldvar = indata.variables[v]
        newvar = DiscreteVariable(oldvar.name, numbins)
        newvar.__dict__.update(oldvar.__dict__) # copy any other data attached to variable
        newvar.arity = numbins
        indata.variables[v] = newvar

    # if discretized all variables, then cast observations to int
    if len(includevars) == indata.variables.size:
        indata.observations = indata.observations.astype(int)
    
    return indata

#
# Main class for dataset
#
class Dataset(object):
    def __init__(self, observations, missing=None, interventions=None, 
                 variables=None, samples=None, skip_stats=False):
        """Create a pebl Dataset instance.

        A Dataset consists of the following data structures which are all
        numpy.ndarray instances:

        * observations: a 2D matrix of observed values. 
            - dimension 1 is over samples, dimension 2 is over variables.
            - observations[i,j] is the observed value for jth variable in the ith
              sample.
        
        * variables,samples: 1D array of variable or sample annotations
        
        This class provides a few public methods to manipulate datasets; one can
        also use numpy functions/methods directly.

        Required/Default values:

             * The only required argument is observations (a 2D numpy array).
             * If missing or interventions are not specified, they are assumed to
               be all zeros (no missing values and no interventions).
             * If variables or samples are not specified, appropriate Variable or
               Sample annotations are created with only the name attribute.

        Note:
            If you alter Dataset.interventions or Dataset.missing, you must
            call Dataset._calc_stats(). This is a terrible hack but it speeds
            up pebl when used with datasets without interventions or missing
            values (a common case).

        """

        self.observations = observations
        self.variables = variables
        self.samples = samples

        # With a numpy array X, we can't do 'if not X' to check the
        # truth value because it raises an exception. So, we must use the
        # non-pythonic 'if X is None'
        
        obs = observations
        if variables is None:
            self.variables = N.array([Variable(str(i)) for i in xrange(obs.shape[1])])
            self._guess_arities()
        if samples is None:
            self.samples = N.array([Sample(str(i)) for i in xrange(obs.shape[0])])


    # 
    # public methods
    # 
    def subset(self, variables=None, samples=None):
        """Returns a subset of the dataset (and metadata).
        
        Specify the variables and samples for creating a subset of the data.
        variables and samples should be a list of ids. If not specified, it is
        assumed to be all variables or samples. 

        Some examples:
        
            - d.subset([3], [4])
            - d.subset([3,1,2])
            - d.subset(samples=[5,2,7,1])
        
        Note: order matters! d.subset([3,1,2]) != d.subset([1,2,3])

        """

        variables = variables if variables is not None else range(self.variables.size)
        samples = samples if samples is not None else range(self.samples.size)
        skip_stats = True
        d = Dataset(
            self.observations[N.ix_(samples,variables)],
            self.missing[N.ix_(samples,variables)],
            self.interventions[N.ix_(samples,variables)],
            self.variables[variables],
            self.samples[samples],
            skip_stats = skip_stats
        )
        
        return d

    
    def _subset_ni_fast(self, variables):
        ds = _FastDataset.__new__(_FastDataset)

        ds.observations = self.observations[:,variables]
        ds.samples = self.samples

        ds.variables = self.variables[variables]
        return ds



    def discretize(self, includevars=None, excludevars=[], numbins=3):
        """Discretize (or bin) the data in-place.

        This method is just an alias for pebl.discretizer.maximum_entropy_discretizer()
        See the module documentation for pebl.discretizer for more information.

        """
        self.original_observations = self.observations.copy()
        self = maximum_entropy_discretize(
           self, 
           includevars, excludevars, 
           numbins
        ) 


    def tofile(self, filename, *args, **kwargs):
        """Write the data and metadata to file in a tab-delimited format."""
        
        with file(filename, 'w') as f:
            f.write(self.tostring(*args, **kwargs))


    def tostring(self, linesep='\n', variable_header=True, sample_header=True):
        """Return the data and metadata as a string in a tab-delimited format.
        
        If variable_header is True, include variable names and type.
        If sample_header is True, include sample names.
        Both are True by default.

        """

        
        def variable(v):
            name = v.name

            if isinstance(v, ClassVariable):
                return "%s,class(%s)" % (name, ','.join(v.labels))    
            elif isinstance(v, DiscreteVariable):
                return "%s,discrete(%d)" % (name, v.arity)
            elif isinstance(v, ContinuousVariable):
                return "%s,continuous" % name
            else:
                return v.name

        # ---------------------------------------------------------------------

        # python strings are immutable, so string concatenation is expensive!
        # preferred way is to make list of lines, then use one join.
        lines = []

        # add variable annotations
        if sample_header:
            lines.append("\t".join([variable(v) for v in self.variables]))
        
        # format data
        nrows,ncols = self.shape
        d = [[(r,c) for c in xrange(ncols)] for r in xrange(nrows)]
        
        # add sample names if we have them
        if sample_header and hasattr(self.samples[0], 'name'):
            d = [[s.name] + row for row,s in zip(d,self.samples)]

        # add data to lines
        lines.extend(["\t".join(row) for row in d])
        
        return linesep.join(lines)


    #
    # private methods/properties
    #


class _FastDataset(Dataset):
    """A version of the Dataset class created by the _subset_ni_fast method.

    The Dataset._subset_ni_fast method creates a quick and dirty subset that
    skips many steps. It's a private method used by the evaluator module. Do
    not use this unless you know what you're doing.  
    
    """
    pass


#
# Factory Functions
#
def fromfile(filename):
    """Parse file and return a Dataset instance.

    The data file is expected to conform to the following format

        - comment lines begin with '#' and are ignored.
        - The first non-comment line *must* specify variable annotations
          separated by tab characters.
        - data lines specify the data values separated by tab characters.
        - data lines *can* include sample names
    
    A data value specifies the observed numeric value, whether it's missing and
    whether it represents an intervention:

        - An 'x' or 'X' indicate that the value is missing
        - A '!' before or after the numeric value indicates an intervention

    Variable annotations specify the name of the variable and, *optionally*,
    the data type.

    Examples include:

        - Foo                     : just variable name
        - Foo,continuous          : Foo is a continuous variable
        - Foo,discrete(3)         : Foo is a discrete variable with arity of 3
        - Foo,class(normal,cancer): Foo is a class variable with arity of 2 and
                                    values of either normal or cancer.

    """
    
    with file(filename) as f:
        return fromstring(f.read())


def fromstring(stringrep, fieldsep='\t'):
    """Parse the string representation of a dataset and return a Dataset instance.
    
    See the documentation for fromfile() for information about file format.
    
    """

    # parse a data item (examples: '5' '2.5', 'X', 'X!', '5!')
    def dataitem(item, v):
        item = item.strip()

        intervention = False
        missing = False
        

        # convert to expected data type
        val = item
        if isinstance(v, ClassVariable):
            try:
                val = v.label2int[val]
            except KeyError:
                raise ClassVariableError()

        elif isinstance(v, DiscreteVariable):
            try:
                val = int(val)
            except ValueError:
                msg = "Invalid value for discrete variable %s: %s" % (v.name, val)
                raise ParsingError(msg)

        elif isinstance(v, ContinuousVariable):
            try:
                val = float(val)
            except ValueError:
                msg = "Invalid value for continuous variable %s: %s" % (v.name, val)
                raise ParsingError(msg)
        else:
            # if not specified, try parsing as float or int
            if '.' in val:
                try:
                    val = float(val)
                except:
                    msg = "Cannot convert value %s to a float." % val
                    raise ParsingError(msg)
            else:
                try:
                    val = int(val)
                except:
                    msg = "Cannot convert value %s to an int." % val
                    raise ParsingError(msg)

        return (val, missing, intervention)


    dtype_re = re.compile("([\w\d_-]+)[\(]*([\w\d\s,]*)[\)]*") 
    def variable(v):
        # MS Excel encloses cells with punctuations in double quotes 
        # and many people use Excel to prepare data
        v = v.strip("\"")

        parts = v.split(",", 1)
        if len(parts) is 2:  # datatype specified?
            name,dtype = parts
            match = dtype_re.match(dtype)
            if not match:
                raise ParsingError("Error parsing variable header: %s" % v)
            dtype_name,dtype_param = match.groups()
            dtype_name = dtype_name.lower()
        else:
            name = parts[0]
            dtype_name, dtype_param = None,None

        vartypes = {
            None: Variable,
            'continuous': ContinuousVariable,
            'discrete': DiscreteVariable,
            'class': ClassVariable
        }
        
        return vartypes[dtype_name](name, dtype_param)

    # -------------------------------------------------------------------------

    # split on all known line seperators, ignoring blank and comment lines
    lines = (l.strip() for l in stringrep.splitlines() if l)
    lines = (l for l in lines if not l.startswith('#'))
    
    # parse variable annotations (first non-comment line)
    variables = lines.next().split(fieldsep)
    variables = N.array([variable(v) for v in variables])

    # split data into cells
    d = [[c for c in row.split(fieldsep)] for row in lines]

    # does file contain sample names?
    samplenames = True if len(d[0]) == len(variables) + 1 else False
    samples = None
    if samplenames:
        samples = N.array([Sample(row[0]) for row in d])
        d = [row[1:] for row in d]
    
    # parse data lines and separate into 3 numpy arrays
    #    d is a 3D array where the inner dimension is over 
    #    (values, missing, interventions) transpose(2,0,1) makes the inner
    #    dimension the outer one
    d = N.array([[dataitem(c,v) for c,v in zip(row,variables)] for row in d]) 
    obs, missing, interventions = d.transpose(2,0,1)

    # pack observations into bytes if possible (they're integers and < 255)
    dtype = 'int' if obs.dtype.kind is 'i' else obs.dtype
    
    # x.astype() returns a casted *copy* of x
    # returning copies of observations, missing and interventions ensures that
    # they're contiguous in memory (should speedup future calculations)
    d = Dataset(
        obs.astype(dtype),
        missing.astype(bool),
        interventions.astype(bool), 
        variables, 
        samples,
    )
    return d
