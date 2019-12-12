"""Example of a devito forward/gradient implementation for a single source with odl."""

import numpy as np
import odl
from devito import Function, TimeFunction, Grid
from examples.seismic import Model, RickerSource, Receiver, TimeAxis, PointSource
from examples.seismic.acoustic import AcousticWaveSolver
import copy

__all__ = ['F', 'FT']


class DevitoSet(odl.set.Set):

    """Set with no member elements (except ``None``).
    ``None`` is considered as "no element", i.e. ``None in EmptySet()``
    is the only test that evaluates to ``True``.
    """
    def __init__(self, devito_in):
        self.input = devito_in
        self.name = devito_in.name

    def __contains__(self, other):
        """Return ``other in self``, always ``False`` except for ``None``."""
        if isinstance(self.input, Function):
            return isinstance(other, Function)
        elif isinstance(self.input, PointSource):
            return isinstance(other, PointSource)

    def contains_set(self, other):
        """Return ``True`` for the empty set, ``False`` otherwise."""
        return isinstance(other, odl.EmptySet)

    def __eq__(self, other):
        """Return ``self == other``."""
        return self.input == other.input

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

    # @property
    # def norm(self):
    #     return np.linalg.norm(self.input)

    @property
    def shape(self):
        return self.input.shape

    @property
    def dtype(self):
        return self.input.dtype

    def element(self, inp=None):
        """Return an element from ``inp`` or from scratch.
        This method should be overridden by subclasses.
        """
        if isinstance(self.input, Function):
            # from IPython import embed; embed()
            new = Function(name=self.name+'_out', grid=self.input.grid, space_order=self.input.space_order)
        elif isinstance(self.input, PointSource):
            new = PointSource(name=self.name+'_new', grid=self.input.grid, coordinates=self.input.coordinates.data[:],
                              time_range=self.input.time_range, npoint=self.input.npoint)
        if inp is not None:
            new.data[:] = inp
        return new

class F(odl.Operator):
    def __init__(self, model, src, rec_geom, time_start, time_end, space_order=16, **kwargs):
        self.data = Receiver(name='rec', grid=model.grid, npoint=rec_geom.shape[0],
                             time_range=src.time_range,
                             coordinates=rec_geom)
        self.source =  Receiver(name='src', grid=model.grid, npoint=src.npoint,
                                time_range=src.time_range,
                                coordinates=src.coordinates.data)
        self.source.data[:] = src.data[:]
        self.space_order = space_order
        self.kwargs = kwargs
        self.model = model
        self.time_start = time_start
        self.time_end = time_end
        self.space_order = space_order   
        self.solver = AcousticWaveSolver(model, source=self.source, receiver=self.data,
                                         space_order=space_order, **kwargs)
       
        self.grid = copy.copy(self.model.grid)
        self.grid.shape = (self.model.grid.shape[0]*3, self.model.grid.shape[1])

        u = Function(name="u", grid=self.grid, space_order=space_order)

        domain = DevitoSet(u)
        im = DevitoSet(u)
        super(F, self).__init__(domain=domain, range=im)
        self.FT  = FT(self.model, self.source,
                      self.data.coordinates.data,
                      self.data.data, self.time_start, self.time_end,
                      space_order=self.space_order,
                      **self.kwargs)

    def _call(self, x, out=None):
 
        u_data = copy.copy(x.data.reshape((3, self.model.grid.shape[0], self.model.grid.shape[1])))
        u_in = TimeFunction(name="u_in", grid=self.model.grid, time_order=2, space_order=self.space_order)
        u_in.data[:] = u_data

        self.solver.forward(m=self.model.m, src=self.source, time_m=self.time_start, time=self.time_end, 
            u=u_in)

        if out is not None:
            out.data[:] = u_in.data.reshape(1, self.grid.shape[0], self.grid.shape[1])
        else:
            out = Function(name="u_out", grid=self.grid, space_order=self.space_order)
            out.data[:] = u_in.data.reshape(1, self.grid.shape[0], self.grid.shape[1])

        return out

    @property
    def adjoint(self):
        # from IPython import embed; embed()
        return self.FT
    
    def derivative(self, x):
        return self

    def opnorm(self):
        return 1


class FT(odl.Operator):
    def __init__(self, model, src, rec_geom, rec_data, time_start, time_end, space_order=16, **kwargs):
        self.data = Receiver(name='rec', grid=model.grid, npoint=rec_geom.shape[0],
                             time_range=src.time_range,
                             coordinates=rec_geom)
        self.data.data[:] = rec_data
        self.source =  Receiver(name='src', grid=model.grid, npoint=src.npoint,
                                time_range=src.time_range,
                                coordinates=src.coordinates.data)
        self.source.data[:] = src.data[:]
        self.space_order = space_order
        self.kwargs = kwargs
        self.model = model
        self.time_start = time_start
        self.time_end = time_end
        self.solver = AcousticWaveSolver(model, source=self.source, receiver=self.data, 
                                            space_order=space_order, **kwargs)
       
        self.grid = copy.copy(self.model.grid)
        self.grid.shape = (self.model.grid.shape[0]*3, self.model.grid.shape[1])
        u = Function(name="u", grid=self.grid, space_order=space_order)

        domain = DevitoSet(u)
        im = DevitoSet(u)
        super(FT, self).__init__(domain=domain, range=im)

    def _call(self, x, out=None):

        u_data = copy.copy(x.data.reshape((3, self.model.grid.shape[0], self.model.grid.shape[1])))
        u_in = TimeFunction(name="u_in", grid=self.model.grid, time_order=2, space_order=self.space_order)
        u_in.data[:] = u_data

        self.solver.adjoint(m=self.model.m, rec=self.data, time_m=self.time_start, time=self.time_end, v=u_in)

        if out is not None:
            out.data[:] = u_in.data.reshape(1, self.grid.shape[0], self.grid.shape[1])
        else:
            out = Function(name="u_out", grid=self.grid, space_order=self.space_order)
            out.data[:] = u_in.data.reshape(1, self.grid.shape[0], self.grid.shape[1])

        return out


    @property
    def adjoint(self):
        return F(self.model, self.source,
                 self.data.coordinates.data,
                 space_order=self.space_order,
                 **self.kwargs)

    def opnorm(self):
        return 1

