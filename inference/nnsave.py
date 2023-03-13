"""
Save torch and tensorflow models using pickle, the same way you would a
scikit-learn model. Just replace pickle.dump with nnsave.dump and you are
golden.
"""
# LEAVE THIS LINE EMPTY. USED FOR META-PROGRAMMING.
def _PyObject_Call(f, args, kwargs):
  return f(*args, **kwargs)

def _new_tf_layer(layer_type, config, weights):
  layer = layer_type.from_config(config)
  layer.set_weights(weights)
  return layer

try:
  from nnsave import _PyObject_Call, _new_tf_layer
except:
  pass
# END METAPROGRAM SUBSET

import bz2
import collections
import configparser
import contextlib
import copyreg
import ensurepip
import gzip
import io
import importlib.abc
import importlib.machinery
importlib_metadata = None
import inspect
import itertools
import linecache
import lzma
import mimetypes
import operator
import os.path
import pathlib
import pickle
import pickletools
import pkgutil
import re
import tempfile
import threading
tomllib = None
import types
import typing
import subprocess
import sys
import uuid
import warnings
import zipfile

_loaded_extra_dispatch_table = False
np = None
torch = None
tf = None


def _load_backports():
  global importlib_metadata

  if sys.hexversion >= 0x30a00f0:
    import importlib.metadata as importlib_metadata
  else:
    try:
      import importlib_metadata # install importlib-metadata
    except:
      importlib_metadata = None

  global tomllib

  if sys.hexversion >= 0x30b00f0:
    import tomllib
  else:
    try:
      import tomli as tomllib
    except:
      tomllib = None # You should install "tomli", but not 100% important.

_load_backports()

def load_extra_dispatch_table():
  _load_backports()

  global _loaded_extra_dispatch_table
  _loaded_extra_dispatch_table = True

  global np
  try:
    import numpy as np
  except:
    pass

  global torch
  try:
    import torch
  except:
    pass

  global tf
  try:
    import tensorflow as tf
  except:
    pass

  if torch is not None:
    _extra_dispatch_table.update({
      tensor_type: _torch_reduce_via_numpy
      for module in (torch, torch.cuda)
      for tensor_type in {
        getattr(module, n) for n in dir(module) if n.endswith('Tensor')
      }
    })

  if tf is not None:
    _TFEagerTensor = type(tf.constant([[1,2,3],[4,5,6]]))
    _extra_dispatch_table[tf.Tensor] = _tf_reduce_via_numpy
    _extra_dispatch_table[_TFEagerTensor] = _tf_reduce_via_numpy
    _extra_dispatch_table[tf.RaggedTensor] = _tf_ragged_reduce_via_numpy
    _extra_dispatch_table[tf.sparse.SparseTensor] = _tf_sparse_reduce_via_numpy


# Additional reduction functions for torch and tensorflow to save using numpy

def _torch_reduce_via_numpy(tensor):
  # Handle named tensors, sparse tensors
  tensor = tensor.detach().cpu()

  if tensor.layout is torch.strided:
    is_named = any(n is not None for n in tensor.names)
    reduction = (torch.tensor, (tensor.numpy(),))
    if is_named:
      return _ef_PyObject_Call, (*reduction, {'names': tensor.names})
    else:
      return reduction
  elif tensor.layout is torch.sparse_coo:
    return (torch.sparse_coo_tensor, (
      tensor._indices().numpy(),
      tensor._values().numpy(),
      tensor.shape
    ))
  elif tensor.layout is torch.sparse_csr:
    return (torch.sparse_csr_tensor, (
      tensor.crow_indices().numpy(),
      tensor.col_indices().numpy(),
      tensor.values().numpy(),
      tensor.shape
    ))
  else:
    raise NotSupportedError()

def _tf_reduce_via_numpy(tensor):
  return (tf.constant, (tensor.numpy(),))

def _tf_ragged_reduce_via_numpy(tensor):
  return (tf.RaggedTensor.from_row_lengths, (tensor.flat_values, tensor.row_lengths()))

def _tf_sparse_reduce_via_numpy(tensor):
  return (tf.sparse.SparseTensor, (tensor.indices, tensor.values, tensor.dense_shape))

def _tf_reduce_layer(layer):
  config = layer.get_config()
  weights = layer.get_weights()
  return (_ef_new_tf_layer, (type(layer), config, weights))


# Modify pickle.Pickler instances to support torch and tensorflow
# Can be run globally by running fix_dispatch_table() or locally on just
# a handful of picklers by running fix_dispatch_table(pickler).
# pickle.dump(s)|load(s) counterparts are available.

class ExtraDispatchTableWrapper(collections.ChainMap):
  __slots__ = ()

class ExtraDispatchTable(dict):
  __slots__ = ()
  def __missing__(self, key):
    if tf and issubclass(key, tf.keras.layers.Layer):
      return _tf_reduce_layer
    raise KeyError(key)

_extra_dispatch_table = ExtraDispatchTable()
extra_dispatch_table = types.MappingProxyType(_extra_dispatch_table)

def fix_dispatch_table(pickler=copyreg):
  if not _loaded_extra_dispatch_table:
    load_extra_dispatch_table()
  if getattr(pickler, 'dispatch_table', None) is None:
    pickler.dispatch_table = copyreg.dispatch_table.copy()
  pickler.dispatch_table = ExtraDispatchTableWrapper(
    pickler.dispatch_table,
    extra_dispatch_table
  )

def unfix_dispatch_table(pickler=copyreg):
  if getattr(pickler, 'dispatch_table', None) is not None:
    if isinstance(pickler.dispatch_table, ExtraDispatchTableWrapper):
      pickler.dispatch_table = pickler.dispatch_table.maps[0]

def dumps(obj, *args, **kwargs):
  file = io.BytesIO()
  pickler = pickle.Pickler(file, *args, **kwargs)
  fix_dispatch_table(pickler)
  pickler.dump(obj)
  file.seek(0)
  return file.read()

def dump(obj, file, *args, **kwargs):
  pickler = pickle.Pickler(file, *args, **kwargs)
  fix_dispatch_table(pickler)
  pickler.dump(obj)

def dumpgz(obj, filename, compresslevel=9, *args, gzmodule=gzip, **kwargs):
  with gzmodule.open(filename, 'wb', compresslevel) as f:
    dump(obj, f, *args, **kwargs)

def loadgz(filename, compresslevel=9, *args, gzmodule=gzip, **kwargs):
  with gzmodule.open(filename, 'rb', compresslevel) as f:
    return pickle.load(f, *args, **kwargs)

def _test_magic(file, mime, magic):
  if isinstance(file, (os.PathLike, str)):
    if mimetypes.guess_type(file)[1] == mime:
      return True
    with open(file, 'rb') as f:
      return f.read(len(magic))
  f.seek(0)
  return f.read(len(magic))

def load(file: typing.Union[os.PathLike, str, bytes]):
  """
  A master load function that can handle gzip-ed pickles, zip files with a
  "model.pkl" inside, or just a plain old pickle. If you know which one you
  have, just use the more specialized function that exists.

  Only use with a trusted model file. This function will run whatever code is
  inside of that model file and that code could be dangerous. When using this
  function, use the same caution that you would use when running a random
  script in a random Git repo you found.
  """
  if isinstance(file, bytes):
    file = io.BytesIO(file)
  pathlike = (os.PathLike, str)
  is_zip = zipfile.is_zipfile(file)
  is_dir = os.path.isdir(file)
  if is_zip or is_dir:
    assert isinstance(file, pathlike), "Can only handle zip files that are not in memory."
    if is_zip and np is not None:
      # Check to see if numpy.
      try:
        return np.load(file)
      except:
        pass
    if is_zip and torch is not None:
      # Maybe a PyTorch model. Need a better way to check.
      try:
        return torch.load(file)
      except:
        pass
    if is_dir and tf is not None:
      # Maybe a Tensorflow model. Need a better way to check.
      try:
        return tf.keras.models.load_model(file)
      except:
        pass
    # TODO: Make this better. Only this case REALLY matters, but this function is not great.
    with PackageSandbox(str(file)) as sand:
      missing_reqs = sand.get_missing_requirements()
      assert len(missing_reqs) == 0, f"Missing required packages: {' '.join(missing_reqs)}"
      try:
        entry_point = sand.load_entry_point()
      except:
        raise
        entry_point = None
      if entry_point is None:
        return sand.load_pickle()
      else:
        return entry_point()
  elif _test_magic(file, None, b'\x93NUMPY'):
    return np.load(file, allow_pickle=True)
  elif _test_magic(file, 'gzip', b'\x1f\x8b'):
    return loadgz(file)
  elif _test_magic(file, 'bzip2', b'BZ'):
    return loadgz(file, gzmodule=bz2)
  elif _test_magic(file, 'xz', b'\xFD7zXZ\0'):
    return loadgz(file, gzmodule=lzma)
  else:
    # Guess it is a pickle
    if isinstance(file, bytes):
      return pickle.loads(file)
    elif isinstance(file, pathlike):
      with open(file, 'rb') as f:
        return pickle.load(f)
    else:
      return pickle.load(file)

# https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html

# CODE FROM DILL: https://github.com/uqfoundation/dill/blob/master/dill/_shims.py
class Reduce(object):
    """
    Reduce objects are wrappers used for compatibility enforcement during
    unpickle-time. They should only be used in calls to pickler.save and
    other Reduce objects. They are only evaluated within unpickler.load.

    Pickling a Reduce object makes the two implementations equivalent:

    pickler.save(Reduce(*reduction))

    pickler.save_reduce(*reduction, obj=reduction)
    """
    __slots__ = ['reduction']
    def __new__(cls, *reduction, **kwargs):
        """
        Args:
            *reduction: a tuple that matches the format given here:
              https://docs.python.org/3/library/pickle.html#object.__reduce__
            is_callable: a bool to indicate that the object created by
              unpickling `reduction` is callable. If true, the current Reduce
              is allowed to be used as the function in further save_reduce calls
              or Reduce objects.
        """
        is_callable = kwargs.get('is_callable', False) # Pleases Py2. Can be removed later
        if is_callable:
            self = object.__new__(_CallableReduce)
        else:
            self = object.__new__(Reduce)
        self.reduction = reduction
        return self
    def __repr__(self):
        return 'Reduce%s' % (self.reduction,)
    def __copy__(self):
        return self # pragma: no cover
    def __deepcopy__(self, memo):
        return self # pragma: no cover
    def __reduce__(self):
        return self.reduction
    def __reduce_ex__(self, protocol):
        return self.__reduce__()

class _CallableReduce(Reduce):
    # A version of Reduce for functions. Used to trick pickler.save_reduce into
    # thinking that Reduce objects of functions are themselves meaningful functions.
    def __call__(self, *args, **kwargs):
        reduction = self.__reduce__()
        func = reduction[0]
        f_args = reduction[1]
        obj = func(*f_args)
        return obj(*args, **kwargs)
# END CODE FROM DILL

# Create functions apply and _new_tf_layer for use in pickles where this
# library is not present.
_thunk = __loader__.get_source(__name__) \
  .split('# LEAVE THIS LINE EMPTY. USED FOR META-PROGRAMMING.', 1)[1] \
  .split("# END METAPROGRAM SUBSET", 1)[0]

_globs = {}
_globs['_thunk'] = Reduce(exec, (_thunk, _globs, _globs))
_globs_created = _globs.copy()
exec(_thunk, _globs_created, _globs_created)

class _EmbeddedFunction:
  """
  Represents a function that needs to be constructed in the pickle in the event
  that this library is not available at unpickling.
  """
  __slots__ = ('func_name')
  def __init__(self, func_name):
    self.func_name = func_name
  def __call__(self, *args, **kwargs):
    return _globs_created[self.func_name](*args, **kwargs)
  def __reduce__(self):
    return operator.itemgetter(self.func_name), (_globs,)

def _set_nnsave_requirement(require=True):
  global _ef_PyObject_Call, _ef_new_tf_layer
  if require:
    _ef_PyObject_Call = _PyObject_Call
    _ef_new_tf_layer = _new_tf_layer
  else:
    _ef_PyObject_Call = _EmbeddedFunction("_PyObject_Call")
    _ef_new_tf_layer = _EmbeddedFunction("_new_tf_layer")

_set_nnsave_requirement(False)

# Sandbox models into their own namespaces to avoid punning, similar to what
# torch.package does.

def _get_requirement_names(reqs):
  return {
    re.split('[><=]', req, 1)[0].strip()
    for req in reqs
  }

class PackageScope(contextlib.AbstractContextManager):
  __slots__ = ('_location',)
  def __init__(self, location):
    assert location is not None
    self._location = location

  def __enter__(self):
    sys.path.insert(0, self._location)
    return self

  def __exit__(self, exc_type, exc, exc_tb):
    sys.path.remove(self._location)

  def _load_data(self, name, strtype, from_file, from_str):
    loader = pkgutil.get_importer(self._location)
    if isinstance(loader, importlib.machinery.FileFinder):
      with open(os.path.join(loader.path, name), 'r' if issubclass(strtype, str) else 'rb') as f:
        return from_file(f)
    elif hasattr(loader, 'get_data'):
      as_str = loader.get_data(name)
      if issubclass(trtype, str):
        as_str = as_str.decode()
      return from_str(as_str)
    else:
      raise NotImplementedError("Cannot get data for loader %s." % (loader,))

  def load_pickle(self, name=None, pickle_module=pickle):
    if name is None:
      pyproject = self.load_pyproject_toml()
      name = 'model.pkl'
      if pyproject is not None:
        try:
          name = pyproject['tool']['nnsave']['model_pickle']
        except KeyError:
          pass
    return self._load_data(name, bytes, pickle_module.load, pickle_module.loads)
    # loader = pkgutil.get_importer(self._location)
    # if isinstance(loader, importlib.machinery.FileFinder):
    #   with open(os.path.join(loader.path, name), 'rb') as f:
    #     return pickle_module.load(f)
    # elif hasattr(loader, 'get_data'):
    #   return pickle_module.loads(loader.get_data(name))
    # else:
    #   raise NotImplementedError("Cannot get data for loader %s." % (loader,))

  def load_pyproject_toml(self):
    try:
      return self._load_data('pyproject.toml', bytes, tomllib.load, tomllib.loads)
    except:
      if tomllib is None:
        try:
          # Does the file 'pyproject.toml' exist?
          self._load_data('pyproject.toml', bytes, lambda f: None, lambda s: None)
        except:
          pass
        else:
          warnings.warn('Install "tomli" to use "pyproject.toml" to specify the requirements.', stacklevel=2)
      return None

  def load_requirements(self):
    pyproject = self.load_pyproject_toml()
    # TODO: Get requirements from wheel

    if pyproject is not None:
      requirements = None

      try:
        requirements = pyproject['project']['dependencies']
      except KeyError:
        pass

      try:
        requirements = pyproject['tool']['poetry']['dependencies']
      except KeyError:
        pass

      if requirements is not None:
        # TODO: Handle dicts; if not isinstance()
        return requirements

    return self._load_data('requirements.txt', str,
      lambda f: [l.strip() for l in f.readlines()],
      lambda s: [l for l in map(str.strip, s.split('\n')) if l]
    )

  def load_entry_point(self, entry_point=None):
    try:
      entry_points_cfg = self.load_pyproject_toml()['project']['entry-points']
    except:
      zf = zipfile.ZipFile(self._location)
      zname = [n for n in zf.namelist() if n.endswith('.egg-info/entry_points.txt')]
      assert len(zname) == 1, 'Multiple .egg-info folders found.'
      zname = zname[0]
      entry_points_cfg = configparser.ConfigParser()
      entry_points_cfg.read(zname)
    load_model = entry_points_cfg['nnsave.load_model']
    if entry_point is None:
      assert len(load_model) == 1, '[project.entry-points."nnsave.load_model"] must only have one entry point.'
      entry_point = next(iter(load_model.keys()))
    return pkgutil.resolve_name(load_model[entry_point])

  def get_missing_requirements(self):
    reqs = self.load_requirements()
    reqs = _get_requirement_names(reqs)
    available = importlib_metadata.packages_distributions()
    available = set(itertools.chain.from_iterable(available.values()))
    return reqs - available

  def install_requirements(self):
    if importlib_metadata is None or self.get_missing_requirements():
      # Load linecache of this module, so if it is reinstalled, error messages
      # have the correct information.
      linecache.getline(__file__, 1, __dict__)

      reqs = self.load_requirements()
      try:
        import pip
      except:
        ensurepip.bootstrap()
      with tempfile.NamedTemporaryFile() as f:
        f.writelines(reqs)
        if importlib_metadata is None:
          f.writeline('importlib_metadata')
        subprocess.run([sys.executable or 'python3', '-m', 'pip', 'install', '-r', f.name])
      # TODO: If torch/tensorflow or any package were reinstalled after loading
      # (except this one), funky things will happen. Do not do that. Be safe
      # and install_requirements in __main__ BEFORE anything else, and then
      # import any packages you like.

  @property
  def location(self):
    return self._location


class PackageSandbox(PackageScope):
  __slots__ = ('_sandbox', '_prefix', '_package_names', '_packages', '_PackageSandboxMetaFinder')
  _CURRENT_SANDBOX : typing.ClassVar['PackageSandbox'] = None
  _LOCK: typing.ClassVar[threading.Lock] = threading.Lock()
  def __init__(self, path, prefix=None):
    super().__init__(path)
    if prefix is None:
      prefix = "_" + uuid.uuid4().hex

    self._sandbox = _sandbox = set()
    self._prefix = prefix
    self._package_names = [spec.name for spec in pkgutil.walk_packages([path])]
    self._packages = {} # cache of the packages that are already imported that conflict in name

    class _PackageSandboxMetaFinder(importlib.abc.MetaPathFinder):
      __slots__ = ()
      @classmethod
      def find_spec(cls, fullname, path, target=None):
        nonlocal _sandbox
        _sandbox.add(fullname)
        # print(cls, fullname, path, target, _sandbox)

    self._PackageSandboxMetaFinder = _PackageSandboxMetaFinder

  def __enter__(self):
    with self._LOCK:
      if self._CURRENT_SANDBOX is not None:
        self._CURRENT_SANDBOX.__exit__(None, None, None)
        self._CURRENT_SANDBOX = None
      for k in self._package_names:
        self._packages[k] = sys.modules.pop(k, None)
      for k in self._sandbox:
        v = sys.modules.pop(self._prefix + '.' + k, None)
        if v is None:
          continue
        # Optional 3 lines
        v.__name__ = k
        if getattr(v, '__spec__', None) is not None:
          v.__spec__.name = v.__name__

        sys.modules[v.__name__] = v
        # print(k, v)
      sys.meta_path.insert(0, self._PackageSandboxMetaFinder)
      sys.path.insert(0, self._location)
      self._CURRENT_SANDBOX = self
    return self

  def __exit__(self, exc_type, exc, exc_tb):
    with self._LOCK:
      assert self._CURRENT_SANDBOX is self
      sys.path.remove(self._location)
      sys.meta_path.remove(self._PackageSandboxMetaFinder)
      absprefix = os.path.abspath(self._prefix)
      self._sandbox = {k for k in self._sandbox if k in sys.modules and not
        # Is module.__file__ inside of self._prefix?
        # os.path.relpath(getattr(sys.modules[k], '__file__', '..'), self._prefix)
        # .split('/', 1)[0] == os.path.pardir
        os.path.abspath(getattr(sys.modules[k], '__file__', None) or '/').startswith(absprefix)
      }
      for k in self._sandbox:
        v = sys.modules.pop(k, None)
        if v is None:
          continue
        # Optional 3 lines
        v.__name__ = self._prefix + '.' + k
        if getattr(v, '__spec__', None) is not None:
          v.__spec__.name = v.__name__

        sys.modules[v.__name__] = v
        # Optional for line tracing. Doesn't show line locations for errors that
        # happen during import.
        linecache.getline(v.__file__, 1, vars(v))
        # print(k, v)
      for k, v in self._packages.items():
        if v is not None:
          sys.modules[k] = v

  @property
  def prefix(self):
    return self._prefix


# pickletools.optimize

# TODO: extension load using temporary files


if __name__ == '__main__':
  # with PackageSandbox('.') as sand:
  #   import _dyn_packages
  # print(sys.modules[sand.prefix + '._dyn_packages'])
  # exit()

  # with PackageSandbox('.') as sand: # PackageSandbox
  #   print(sand.get_missing_requirements())
  #   print(sand.load_pickle())
  #   print(sand.load_requirements())
  # print(load('.'))
  # exit()
  load_extra_dispatch_table()

  if torch:
    # f = io.BytesIO()
    buffers = []
    # F = pickle.Pickler(f, protocol=5) #
    # fix_dispatch_table(F)
    x = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
    i = [[0, 1, 1],[2, 0, 2]]
    v =  [3, 4, 5]
    s = torch.sparse_coo_tensor(i, v, (2, 3))
    crow_indices = torch.tensor([0, 2, 4])
    col_indices = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([1, 2, 3, 4])
    csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    named = torch.zeros((2,3), names=('xz', 'y'))
    X = [x, s, csr, named]
    # F.dump(X)
    # f.seek(0)
    # y = pickle.load(f)
    byt = dumps(X, protocol=5, buffer_callback=buffers.append)
    y = pickle.loads(byt, buffers=buffers)
    # f.seek(0)
    pickletools.dis(byt) # f.read()
    print(y)
    print(buffers)
    # import code
    # code.interact(local=locals())
    # with open('model.pkl', 'wb') as f:
    #   f.write(byt)
    # with open('model.dat', 'wb') as f:
    #   raws = [buffer.raw() for buffer in buffers]
    #   f.write(len(raws).to_bytes(8, 'big'))
    #   for raw in raws:
    #     f.write(raw.nbytes.to_bytes(8, 'big'))
    #   for raw in raws:
    #     f.write(raw)
    #   f.flush()
    #   os.fsync(f)
    
    # for buffer in buffers:
      # print(buffer.raw().nbytes)
    
    with open('model.dat', 'rb') as f:
      num_bufs = int.from_bytes(f.read(8), 'big')
      num_bytes = [int.from_bytes(f.read(8), 'big') for _ in range(num_bufs)]
      print(num_bufs, num_bytes)
      import mmap, itertools
      with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
        buffers = [m[i:j] for i, j in itertools.pairwise(itertools.accumulate([8*(1+num_bufs)] + num_bytes))]
        print(buffers)

  if tf:
    class MyModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

      def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

      def get_config(self):
        return {}

    model = MyModel()
    model(tf.zeros((10,10)))
    d = dumps(model)
    pickletools.dis(d)
