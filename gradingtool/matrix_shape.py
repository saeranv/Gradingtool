from __future__ import print_function
from loadenv import *

from matrix_utils import MatrixUtils

class MatrixSurface(MatrixUtils):

    """ Planar Surface
    TODO: delete all the setters
    TODO: delete set private error message
    """

    SET_PRIVATE_ERROR_MSG = "Cannot set variable of property once already set."

    def __init__(self, vertices, osm_surface = None):
        self.vertices = vertices

        # private properties
        self._normal = None
        self._bottom_edges = None
        self._osm_surface = osm_surface
        self._cardinal_orientation = None
        self._edges = None
        self._edge_directions = None
        self._edge_lengths = None
        self._vertices_loop = None
        self._midpts = None
        # TODO: refactor so from AnalysisSurface
        self._space_apts = None
        #self._space_apts_ptindex = None
        #self.space_val_dict = {}
        # f = lambda s: np.array(s.space_val_dict["UDI"])
        # vf = np.vectorize(f, otypes=[object])
        # value_array = vf(glazed_array)

    def __repr__(self):
        s = "MatrixSurface:"
        for v in self.vertices:
            s += "\nv={}".format(v)
        s += "\n"
        return s

    @property
    def bottom_edges(self):
        if self._bottom_edges is None:
            self.bottom_edges = self.group_vector_by_z(self.vertices)
        return self._bottom_edges

    @bottom_edges.setter
    def bottom_edges(self,v):
        self._bottom_edges = v

    @property
    def vertices_loop(self):
        if self._vertices_loop is None:
            loop_len = self.vertices.shape[0] + 1
            loop_array = np.zeros(shape=(loop_len,3))
            for i, pt in enumerate(self.vertices):
                loop_array[i] = self.vertices[i,:]
            loop_array[loop_len-1] = self.vertices[0,:]
            self._vertices_loop = loop_array
        return self._vertices_loop

    @property
    def midpts(self):
        if self._midpts is None:
            self._midpts = [self.midpoint(edge[0], edge[1]) for edge in self.edges]
        return self._midpts

    @property
    def edges(self):
        """ Returns list of edges (start, end pt tuples) """
        if self._edges is None:
            self._edges = self.edges_from_vertices(self.vertices, self.vertices_loop)
        return self._edges

    @property
    def edge_directions(self):
        """ vector directions"""
        if self._edge_directions is None:
            self._edge_directions = [np.linalg.norm(edge[i+1] - edge[i]) for edge in edges]
        return self._edge_directions

    @property
    def edge_lengths(self):
        """ edge lengths """
        if self._edge_lenghts is None:
            self._edge_lengths = [np.linalg.norm(edge[i+1] - edge[i]) for edge in edges]
        return self._edge_lengths

    def get_space_apts(self, loc_lambda, srfi):
        if self._space_apts is None:
            self._space_apts = []
        elif self.is_matrix(self._space_apts):
            if loc_lambda is not None:
                return loc_lambda(self._space_apts, srfi)
            else:
                raise Exception("Must pass a location lambda to access matrix!")
        else:
            return self._apts

    def set_space_apts(self, v):
        if self._space_apts is not None:
            raise Exception(self.SET_PRIVATE_ERROR_MSG)
        self._space_apts = v

    # TODO Wrong chagne
    def insert_space_apts(self, ri, ci, si, srfi, value):
        if self.is_matrix(self._space_apts):
            x = self._space_apts[ri, ci, si][srfi]
            if x is None or x == []:
                self._space_apts[ri, ci, si][srfi] = [value]
            else:
                self._space_apts[ri, ci, si][srfi].append(value)


    @property
    def normal(self):
        if self._normal is None:
            self._normal = self.normal_from_vertices(self.edge_directions)
        return self._normal

    @property
    def space_apts_ptindex(self):
        if self._space_apts_ptindex is None:
            self.space_apts_ptindex = []
        return self._space_apts_ptindex

    @space_apts_ptindex.setter
    def space_apts_ptindex(self, v):
        self._space_apts_ptindex = v

    # TODO: deprecated?
    def load_space_values(self, value_array, value_key):
        self.space_val_dict[value_key] = []

        assert len(self.space_apts) == len(self.space_apts_ptindex)

        for i in range(len(self.space_apts)):
            try:
                value_index = self.space_apts_ptindex[i]
                value_for_point = value_array[value_index]
                self.space_val_dict[value_key].append(value_for_point)
            except:
                pass#print("value length != analysis pt length")

    @property
    def osm_surface(self):
        return self._osm_surface

    @osm_surface.setter
    def osm_surface(self, v):
        if self._osm_surface is not None:
            raise Exception(self.SET_PRIVATE_ERROR_MSG)
        self._osm_surface = v

    @property
    def cardinal_orientation(self):
        if self._cardinal_orientation is None:
            deg = self.get_cardinal_orientation(self.bottom_edges[0])
            self._cardinal_orientation = deg
        return self._cardinal_orientation


class MatrixAnalysisPoint(MatrixUtils):

    """ Analysis Point
        3d Vector, and scalar performance value
    """

    SET_PRIVATE_ERROR_MSG = "Cannot set variable of property once already set."

    def __init__(self, p, v=None):
        self._p = p # 3d vector
        self._v = v if v is not None else [0, 0, 0] # list of scalar performance values

    @property
    def p(self):
        return self._p

    @property
    def v(self):
        return self._v


class MatrixSpace(MatrixUtils):
    """
    Volume composed of MatrixSurfaces
    """

    SET_PRIVATE_ERROR_MSG = "Cannot set variable of property once already set."

    def __init__(self, surfaces, osm_space = None):
        self.surfaces = surfaces

        # private properties
        self._apts = None
        self._osm_space = osm_space

    def get_apts(self, loc_lambda=None):
        if self._apts is None:
            self._apts = []
        elif self.is_matrix(self._apts):
            if loc_lambda is not None:
                return loc_lambda(self._apts)
            else:
                raise Exception("Must pass a location lambda to access matrix!")
        else:
            return self._apts

    def set_apts(self, v):
        if self._apts is not None:
            raise Exception(self.SET_PRIVATE_ERROR_MSG)
        self._apts = v

    @property
    def osm_space(self):
        return self._osm_space

    @osm_space.setter
    def osm_space(self, v):
        if self._osm_space is not None:
            raise Exception(self.SET_PRIVATE_ERROR_MSG)
        self._osm_space = v
