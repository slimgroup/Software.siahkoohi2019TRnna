"""Example of a devito forward/gradient implementation for a single source with odl."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import odl
from devito import Function
from examples.seismic import Model, RickerSource, Receiver, TimeAxis, PointSource
from examples.seismic.acoustic import AcousticWaveSolver, smooth10
from odl_operators import *
from layers_tf_devito import as_tensorflow_layer
import tensorflow as tf
import odl.contrib.tensorflow


# Model
shape = (151, 101)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :] = 1.5
v[:, 51:] = 2.5
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, space_order=1, nbpml=10)
m0 =  Function(name="m0", grid=model.grid, space_order=1)
m0.data[:] = smooth10(model.m.data[:], model.m.data.shape)
dm0 =  Function(name="dm0", grid=model.grid, space_order=1)
dm0.data[:] = -m0.data[:] + model.m.data[:]

# Derive timestepping from model spacing
dt = model.critical_dt
t0 = 0.0
tn = 1000.
time_range = TimeAxis(start=t0, stop=tn, step=dt)
# Source
f0 = 0.015
src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=101, time_range=time_range)
rec_t.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec_t.coordinates.data[:, 1] = 20.


opF = F(model, src, rec_t.coordinates.data)
opJ = opF.derivative(m0)

true_D = opF(model.m)
syn_D = opF(m0)
truc = opJ(dm0)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


# Create tensorflow layer from odl operator
odl_op_layer = as_tensorflow_layer(opJ, 'Born')

# Lazily apply operator in tensorflow

x = tf.constant(np.asarray(dm0.data))
z = tf.constant(np.asarray(true_D.data-syn_D.data))

x_reshaped = x[None, ..., None]
z_reshaped = z[None, ..., None]


y = odl_op_layer(x_reshaped)


# Evaluate using tensorflow
print(y.eval())
# Compare result with pure ODL
print(opJ(x.eval()))

# Evaluate the adjoint of the derivative, called gradient in tensorflow
# We need to scale by cell size to get correct value since the derivative
# in tensorflow uses unweighted spaces.
# scale = ray_transform.range.cell_volume / ray_transform.domain.cell_volume
print(tf.gradients(y, [x_reshaped], z_reshaped)[0].eval())

# Compare result with pure ODL
print(opJ.derivative(x.eval()).adjoint(z.eval()).data)
