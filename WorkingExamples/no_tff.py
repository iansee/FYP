
# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of HDF5 backed ClientData."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import h5py
import six
import tensorflow as tf


class HDF5ClientData():
  """A `tff.simulation.ClientData` backed by an HDF5 file.
  This class expects that the HDF5 file has a top-level group `examples` which
  contains further subgroups, one per user, named by the user ID.
  The `tf.data.Dataset` returned by
  `HDF5ClientData.create_tf_dataset_for_client(client_id)` yields tuples from
  zipping all datasets that were found at `/data/client_id` group, in a similar
  fashion to `tf.data.Dataset.from_tensor_slices()`.
  """

  _EXAMPLES_GROUP = "examples"

  def __init__(self, hdf5_filepath):
    """Constructs a `tff.simulation.ClientData` object.
    Args:
      hdf5_filepath: String path to the hdf5 file.
    """
    self._filepath = hdf5_filepath

    self._h5_file = h5py.File(self._filepath, "r")
    self._client_ids = sorted(
        list(self._h5_file[HDF5ClientData._EXAMPLES_GROUP].keys()))

    # Get the types and shapes from the first client. We do it once during
    # initialization so we can get both properties in one go.
    g = tf.Graph()
    with g.as_default():
      tf_dataset = self._create_dataset(self._client_ids[0])
      self._element_type_structure = tf.data.experimental.get_structure(
          tf_dataset)

  def _create_dataset(self, client_id):
    return tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict((name, ds[()]) for name, ds in sorted(
            six.iteritems(self._h5_file[HDF5ClientData._EXAMPLES_GROUP]
                          [client_id]))))

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    tf_dataset = self._create_dataset(client_id)
    return tf_dataset

  @property
  def element_type_structure(self):
    return self._element_type_structure
