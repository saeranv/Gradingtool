from __future__ import print_function
from loadenv import *

from cache import Cache
from matrix_shape import MatrixAnalysisPoint
from matrix_shape import MatrixSpace
from matrix_shape import MatrixSurface
from matrix_utils import MatrixUtils



def is_near_zero(num, tol=1e-10):
    return num < 1e-10

def project_points_to_line(data_lst, curve, meshpts, DIST_TOL):
    """ for grade plane avg calc """

    def flat_dist_curry(_refpt):
        def flat_dist(p):
            return np.linalg.norm(MatrixUtils.to2d(p) - MatrixUtils.to2d(_refpt))
        return flat_dist


    # Adding path curves to meshpts if exists
    # curve_to_meshpts = data_lst["topocrvs"]
    # if curve_to_meshpts != [] or curve_to_meshpts != [None]:
    #     vertices_to_meshpts = [np.array(v) for v in curve_to_meshpts]
    #     vertices_to_meshpts = MatrixUtils.split_curve_loop_by_distance(vertices_to_meshpts + [vertices_to_meshpts[0]], MIN_SPACING/2.0)
    #     meshpts += vertices_to_meshpts

    # Now normal algo
    # msrftmp = MatrixSurface(np.array([np.array(v) for v in curve]))
    # split_curve = MatrixUtils.split_curve_loop_by_distance(msrftmp.vertices_loop, MIN_SPACING)
    # msrf = MatrixSurface(np.array([np.array(v) for v in split_curve]))

    # print('vertices')
    # pp(msrf.vertices)
    # print(data_lst["note"])
    # we need edges and midpts

    edges = []
    midpts = []
    for i in range(0,len(curve)-2):
        pt1, pt2 = np.array(curve[i]), np.array(curve[i+1])
        #print(pt1, pt2)
        edges.append([pt1,pt2])
        midpts.append(MatrixUtils.midpoint(pt1,pt2))

    #edges = msrf.edges
    #midpts = msrf.midpts

    # Get normal, midpt of each edge
    norms = [MatrixUtils.outside_normal_from_edge(edge) for edge in edges]

    close_pts = [None] * len(edges)
    bool_close_pts = [None] * len(edges) # boolean mask of when there are close_pts or not

    for ei,edge in enumerate(edges):
        #if ei != 3:
        #    continue
        close_pts[ei] = []
        edge_norm = MatrixUtils.to2d(norms[ei])
        edge_midpt = MatrixUtils.to2d(midpts[ei]) #def midpoint(cls, pt1, pt2):

        min_dist_meshpt = float("inf")
        min_meshpt = None

        for mi,meshpt_ in enumerate(meshpts):
            # check direction
            meshpt = MatrixUtils.to2d(np.array(meshpt_))
            mesh_vec = meshpt - edge_midpt
            costheta = np.dot(MatrixUtils.unitize(edge_norm), MatrixUtils.unitize(mesh_vec))

            # check if is perpendicular to edge
            if costheta > 0.0:
                edgept1 = MatrixUtils.to2d(edge[0])
                edgept2 = MatrixUtils.to2d(edge[1])

                dist_y = MatrixUtils.distance_from_line(edgept1, edgept2, meshpt)

                # calc distance x
                max_x_dist = np.linalg.norm(edgept2 - edge_midpt)
                costheta = np.dot(MatrixUtils.unitize(mesh_vec), MatrixUtils.unitize(edgept1 - edge_midpt))
                if 0.0 <= costheta <= 1.0:
                    edge_x_vector = edgept1 - edge_midpt
                else:
                    costheta = np.dot(MatrixUtils.unitize(mesh_vec), MatrixUtils.unitize(edgept2 - edge_midpt))
                    edge_x_vector = edgept2 - edge_midpt

                dist_x = costheta  * np.linalg.norm(mesh_vec)
                within_x_dist = dist_x < max_x_dist
                within_y_dist = dist_y <= DIST_TOL

                if within_x_dist and within_y_dist:
                    if dist_y < min_dist_meshpt:
                        # move meshpt_ along normal back by h
                        proj_vec = -dist_y * norms[ei]
                        proj_pt = proj_vec + np.array(meshpt_)

                        min_dist_meshpt = dist_y
                        min_meshpt = proj_pt
                        #close_pts[ei].append(proj_pt)
                        #ghclose.append(list(meshpt_))
                        #ghclose.append(proj_pt.tolist())

        if min_meshpt is not None:
            close_pts[ei].append(min_meshpt)
            bool_close_pts[ei] = True
        else:
            close_pts[ei].append(np.array(midpts[ei]))
            bool_close_pts[ei] = False

    # flatten and clean list structure
    for ei, _ in enumerate(close_pts):
        refpt = edges[ei][0]
        flat_dist = flat_dist_curry(refpt)

        if bool_close_pts[ei] == True:
            sorted_by_dist = sorted(close_pts[ei], key = flat_dist)
            close_pts[ei] = sorted_by_dist[0]
        else:
            close_pts[ei] = close_pts[ei][0]

    # Now set new hts

    for ei, _ in enumerate(close_pts):
        if bool_close_pts[ei] != True:
            close_pts[ei] = None

    projected_points = [close_pt for close_pt in close_pts if close_pt is not None]
    edge_points_for_dynamo_viz = [p.tolist() for p in projected_points]

    # weighted length average method
    weighted_y = 0
    total_length = 0
    for pi in range(len(projected_points) - 1):
        v1, v2 = projected_points[pi], projected_points[pi + 1]
        edge_vector  = (MatrixUtils.to2d(v2) - MatrixUtils.to2d(v1))

        grade_x = np.linalg.norm(edge_vector)

        slope_vector = (v2 - v1)
        #angle = MatrixUtil.angle(slope_vector, edge_vector)
        crossprod = np.cross(MatrixUtils.unitize(slope_vector), MatrixUtils.unitize(edge_vector))
        sintheta = np.linalg.norm(crossprod)
        grade_y = sintheta * np.linalg.norm(slope_vector)

        #grade_x_array[pi] = grade_x
        #grade_y_array[pi] = grade_y

        weighted_y += v1[2]*grade_x
        total_length += grade_x

    #numerator = np.dot(grade_x_array, grade_y_array)
    #denominator = np.dot(grade_y_array, np.ones(len(projected_points)-1))
    # float(np.mean([pp[2] for pp in projected_points]))
    weighted_avg = float(weighted_y / total_length) #float(numerator/denominator)
    #print(weighted_avg)

    # Send to GH
    midpts = MatrixUtils.makelists(midpts)
    norms = MatrixUtils.makelists(norms)
    closepts = edge_points_for_dynamo_viz


    #D["norm"] = norms
    #D["mpts"] = midpts

    closepts = MatrixUtils.makelists(projected_points)
    number = weighted_avg

    return closepts, number

    print("Calculated!")

if __name__ == "__main__":

    # interface
    interface = Cache()
    interface.set_listeners()

    D = {}
    D["norm"] = []
    D["mpts"] = []
    D["closepts"] = []
    D["number"] = None

    data_lst = interface.recieve()
    curve_pts = data_lst["curves"]
    meshpts = data_lst["points"]
    DIST_TOL = data_lst["perpendicular_dist"] * 1000.0     # 55.0 * 1000.0

    closepts, number = project_points_to_line(data_lst, curve_pts, meshpts, DIST_TOL)
    D["closepts"] += closepts
    D["number"] = number

    interface.send(D)