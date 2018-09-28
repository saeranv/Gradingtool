from __future__ import print_function
from loadenv import *

class MatrixUtils(object):
    """
    This is a classmethod so that we can use as static class,
    or when inherited by MatrixShapes, as class method.
    """

    @classmethod
    def helloworld(cls):
        print("8 test for ipython")

    @classmethod
    def to2d(cls, vec3d):
        return np.array([vec3d[0], vec3d[1]], dtype=float)

    @classmethod
    def is_matrix(cls, v):
        return isinstance(v, np.ndarray)

    @classmethod
    def equalize_list_length(cls, list_of_unequal_lists):
        # for matrix ops
        loul = list_of_unequal_lists
        maxlen = len(sorted(loul, key=len, reverse=True)[0])
        loel = [list + [None] * (maxlen - len(list)) for list in loul]

        return loel

    @classmethod
    def groupby_with_tol(cls, values, keylambda = None, tol=1e-6):
        """ This may be a matrix_util??
        # https://stackoverflow.com/questions/8226923/sort-list-of-floating-point-numbers-in-groups
        """
        keylambda = (lambda v: v) if keylambda==None else keylambda
        values = sorted(values, key = keylambda)
        zvalues = [v[2] for v in values]

        # to understand these operations
        #print(zvalues)
        #print(np.diff(zvalues) > tol)
        #print(np.where(np.diff(zvalues) > tol)[0]+1)

        zgroups = np.split(zvalues, np.where(np.diff(zvalues) > tol)[0]+1)

        grouped_values = [None]*len(zgroups)
        inc_so_far = 0
        for i in range(len(zgroups)):
            inc_to_increase_by = inc_so_far + len(zgroups[i])
            grouped_values[i] = values[inc_so_far:inc_to_increase_by]
            inc_so_far += inc_to_increase_by
            #pp(grouped_values[i])
            #print('---')
        #print('---')

        return grouped_values

    @classmethod
    def group_vector_by_z(cls,vector_lst, tol=1e-6):
        """ Returns [(key, itemlst), (key,itemlst)]
        """
        return cls.groupby_with_tol(vector_lst, keylambda=lambda v:v[2], tol=tol)

    @classmethod
    def distance_from_line(cls, p1, p2, pt):
        """
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        A = 1/2 * b*h
        h = 2A/b
        Find area by taking cross product of two ps with apt

        All points must be 2d!

        """
        edge_vec1 = p1 - pt
        edge_vec2 = p2 - pt

        b = np.linalg.norm(p2 - p1)
        A = abs(np.cross(edge_vec1, edge_vec2)/2.)
        h = 2*A / b


        return h

    @classmethod
    def closest_line_to_point(cls, lines, pt, max_dist = None):
        """ Return line that is closest to pt

            Args:
                lines: a 3d ndarray (array of edges >> array of pts >> array of coordinates)
                    Example structure with two surfaces in the space
                    [
                        [[px,py],[px,py]],  # srf edge1 vertex start and end
                         [px,py],[px,py]]   # srf edge2 vertex start and end
                    ] # space
                    pt: numpy array of 2d coordinates
                    max_dist: optional arg to only look at points in proximity
            Returns:
                min_h: minimum distance from point to line
                min_line_index: row index of the minimum line in lines ndarray. If no points are within max_dist
                    returns None
        """

        # Base cases
        rows = lines.shape[0] #number of lines

        if rows == 0:
            print("No lines in matrix to find closest point")
            return None
        elif rows == 1:
            return 0

        min_h = float("inf")
        min_line_index = None

        for j in range(rows):
            line = lines[j,:]

            # line end pts
            e1 = line[0,:]
            e2 = line[1,:]

            h = cls.distance_from_line(e1, e2, pt)

            if max_dist and h > max_dist:
                continue

            if h < min_h:
                min_h = h
                min_line_index = j

        return min_line_index

    @classmethod
    def group_pts_by_srf(cls, srf_array, apts_in_space, ri, ci, si):
        """ Nest values by surfaces within zone list

            Args:
                srf_lst: A ndarray of MatrixSurfaces per space. Does not have to be all lists (i.e can be filtered)
                pts_in_space: A list of points in the zone.
                val_in_space: A corresponding list of values in the zone
            Returns:
                grouped_by_srf_mtx: m x n x 2 matrix. sparse matrix???
                    m = number of surfaces in space
                    n = sparse vector? = number of closest points to each surface
                    2 = point 2d coordinates
        """
        # Get 2d bottom edge of surface
        srf_edge_mtx = np.array([map(cls.to2d, srf.bottom_edges[0]) for srf in srf_array])

        # Example structure with two surfaces in the space
        # [[[px,py],[px,py]], #edge1 vertex start and end
        #   [px,py],[px,py]]] #edge2 vertex start and end

        for apt_index in range(apts_in_space.shape[0]):

            pt2d = cls.to2d(apts_in_space[apt_index].p)

            # filter out any pt > 5 m
            closest_edge_index = cls.closest_line_to_point(srf_edge_mtx, pt2d, max_dist = 5)

            if closest_edge_index == None:
                continue

            srf_array[closest_edge_index].insert_space_apts(ri, ci, si, closest_edge_index, apts_in_space[apt_index])

        #srf_array[closest_edge_index].space_apts_ptindex.append(pt_index)

        #edge_id_array[pt_index] = closest_edge_index

        #return edge_id_array

    @classmethod
    def unitize(cls, vector):
        #normed_array = np.linalg.norm(vector_array, axis=1, ord=2) #ord = euclidian or square = sum(abs(x)**ord)**(1./ord)
        return vector/np.linalg.norm(vector, ord=2)

    @classmethod
    def degrees(cls, radians):
        return radians * 180./math.pi

    @classmethod
    def edges_from_vertices(cls, vertices, vertices_loop):
        """ Get defining the edges of polygon.
        vertices_loop:  Add first vertice to end
        """

        # 1d array for vector array
        edges = [None] * vertices.shape[0]

        for i in range(vertices.shape[0]):
            edges[i] = (vertices_loop[i], vertices_loop[i+1])

        return edges

    @classmethod
    def normal_from_vertices(cls, direction_vectors):
        """ Returns unit cross product from direction vectors of a planar surface"""
        cross_product_vector = np.cross(direction_vectors[0], direction_vectors[1])
        return cross_product_vector/np.linalg.norm(cross_product_vector)

    def right_or_left(cls, vector1, vector2):
        """
        Ref: https://github.com/saeranv/bibil/blob/master/src/bibil/bibil_shape.py#L210
        Use 2d cross product (axby - bxay) to see if next_vector is right/left
        This requires ccw ordering of vectors
        If cross is positive (for ccw ordering) then next_vector is to left (inner)
        If cross is negative (for ccw ordering) then next_vector is to right (outer)
        If cross is equal then vectors are colinear. Assume inner.
        """
        # TODO: FIX FIX FIX
        dotrad = cls.angle(cls.unitize(vector1), cls.unitize(vector2))
        cross_z_sign = np.cross(cls.unitize(vector1), cls.unitize(vector2))

        if cross_z_sign < 0.0:
                dotrad = 2*math.pi - dotrad
        return dotrad

    @classmethod
    def angle(cls,vector1, vector2):
        """ Returns angle between two vectors in radians """
        # TODO: is error prone. Check this.
        unit_vector1 = vector1
        unit_vector2 = vector2
        # for unit vectors, dot_product gives cos(theta)
        dot_prod = np.dot(unit_vector1, unit_vector2)
        theta = np.arccos(np.clip(dot_prod, -1.0, 1.0)) # clip to avoid numerical tolerance issues
        return theta

    @classmethod
    def group_srf_by_orientation(cls, srf_array, tol=math.pi/2.):
        """ Nest values by surfaces within zone list

            Args:
                srf_array: A 1D array of MatrixSurfaces. Does not have to be all lists (i.e can be filtered)
                tol: angular tolerance in radians. Default is 90.0
            Returns:
                srf_mtx: A ndarray of surfaces split by orientation
        """
        srf_mtx = np.zeros([2,2])
        true_north = np.array([0,1.0,0])
        #v1 = srf_array[0].vertices[1] - srf_array[0].vertices[0]
        #v2 = srf_array[0].vertices[2] - srf_array[0].vertices[1]

        for i in range(srf_array.shape[0]):
            pass
            """
            tn2 = cls.to2d(true_north)
            norm2 = cls.to2d(srf_array[i].normal)
            cross = np.cross(tn2,norm2)

            print(cross)
            #vecposn = "right" if cross > 0.0 else "left"
            #print(vecposn)
            #print(np.linalg.norm(srf_array[i].normal, ord=2))
            print('->', cls.degrees(cls.angle(true_north,srf_array[i].normal)))
            #print(cls.angle(true_north,srf_array[i].normal))
            print(cls.degrees(cls.angle(true_north,srf_array[i].normal)) - 8.93)
            print(cls.degrees(cls.angle(true_north,srf_array[i].normal)) + 8.93)
            print('--')
            """
        # for each space get normal
        # normal
        return srf_mtx

    @classmethod
    def group_srf_by_cardinal_orientation(cls, srf_lst):
        """ Nest values by surfaces within zone list

            Args:
                srf_lst
            Returns:
                [[][][][]]
        """
        n, e, s, w = [], [], [], []

        for i in range(len(srf_lst)):
            srf = srf_lst[i]
            deg = srf.cardinal_orientation

            if deg == 0:
                n.append(srf)
            elif deg == 90:
                e.append(srf)
            elif deg == 180:
                s.append(srf)
            elif deg == 270:
                w.append(srf)
            else:
                raise Exception("error at orientation calc")

        return [n, e, s, w]

    @classmethod
    def group_apts_by_cardinal_orientation(cls, srf_lst, loc_lambda):

        facade_mtx = MatrixUtils.group_srf_by_cardinal_orientation(srf_lst)

        for fi in range(len(facade_mtx)): # 4
            srf_lst = facade_mtx[fi]

            if len(srf_lst) > 0:
                apts_mtx = [srf.get_space_apts(loc_lambda, si) for si, srf in enumerate(srf_lst)]
                facade_mtx[fi] = reduce(lambda a, b: a + b, apts_mtx)

        return facade_mtx

    @classmethod
    def get_cardinal_orientation(self, edge_tuple):
        """
        Return orientation in degrees
        """
        dirvec = edge_tuple[1] - edge_tuple[0]
        dirvec = dirvec/np.linalg.norm(dirvec)

        # Assume edge is CCW
        if np.abs(dirvec[0]) > np.abs(dirvec[1]): # edge is running EW
            if(not is_near_zero(dirvec[0]) and dirvec[0] > 0.0): # edge is running E and facing S
                return 180    # Wall is running E and facing S
            else:
                return 0      # Wall is running W and facing N
        else:
            if(not is_near_zero(dirvec[1]) and dirvec[1] > 0.0):
                return 90     # Wall is running N and facing E
            else:
                return 270    # Wall is running S and facing W

    @classmethod
    def normalize_scalar(cls, val, min, max):
        """
            value increment / magnitude of scalar
            val - min / max - min
        """
        return (val - min) / (max - min)

    @classmethod
    def makelists(cls, L):
        """ recursively check arrays/lists
        """
        if isinstance(L, (list, tuple)):
            return [cls.makelists(subL) for subL in L]
        elif isinstance(L, np.ndarray):
            return [cls.makelists(subL.tolist()) for subL in L]
        else:
            return L

    @classmethod
    def midpoint(cls, pt1, pt2):
        t = np.linalg.norm(pt2 - pt1)/2.
        dirv = cls.unitize((pt2 - pt1))

        return (t * dirv) + pt1

    @classmethod
    def outside_normal_from_edge(cls, edge):
        """
        Assume edge with same z (i.e. bottom edge).
        """
        dirvec = edge[1] - edge[0]
        unitz = np.array([0,0,1])

        # cross product of dirvec, unitz = normal
        norm = MatrixUtils.normal_from_vertices((dirvec, unitz))

        return norm

    @classmethod
    def split_curve_segment_by_distance(cls, v1, v2, min_dist):
        """
        :param v1:
        :param v2:
        :param min_dist:
        :return:
        """
        new_vertices = [v1]

        dist = np.linalg.norm(v2-v1)
        if dist > min_dist:
            subdivs = int(dist/min_dist)
            vdir = cls.unitize(v2-v1)
            for j in range(subdivs):
                inc_dist = (j+1) * min_dist

                if not is_near_zero(inc_dist, eps = 1e-3):
                    new_v = v1 + vdir * inc_dist
                    new_vertices.append(new_v)

        return new_vertices

    @classmethod
    def split_curve_loop_by_distance(cls, vertices_loop, min_dist):
        """ Splits ccw array/list of vertices by a minmumn distance
            Returns a list of vertices in new list
        """
        new_vertices = []

        for i in range(len(vertices_loop)-1):
            v1 = vertices_loop[i]
            v2 = vertices_loop[i+1]

            new_curve_vertices = cls.split_curve_segment_by_distance(v1, v2, min_dist)
            new_vertices.extend(new_curve_vertices)
            # new_vertices.append(v2) # don't need this as new v1 will be same

        new_vertices.pop(-1)

        return new_vertices
