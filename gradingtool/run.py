from __future__ import print_function
from loadenv import *

from cache import Cache
from matrix_shape import MatrixAnalysisPoint
from matrix_shape import MatrixSpace
from matrix_shape import MatrixSurface
from matrix_utils import MatrixUtils


def project_points_to_line(interface):
    """ for grade plane avg calc """

    data_lst = interface.recieve()
    curves = data_lst["curves"]
    meshpts = data_lst["points"]

    curve = curves[0]
    vertices = [np.array(v) for v in curve]
    msrf = MatrixSurface(np.array(vertices))

    # we need edges and midpts
    edges = msrf.edges
    midpts = msrf.midpts

    # Get normal, midpt of each edge
    norms = [MatrixUtils.outside_normal_from_edge(edge) for edge in edges]

    close_pts = [None] * len(edges)

    ghclose = []
    for ei,edge in enumerate(edges):
        #if ei != 3:
        #    continue
        close_pts[ei] = []
        edge_norm = MatrixUtils.to2d(norms[ei])
        edge_midpt = MatrixUtils.to2d(midpts[ei])

        for mi,meshpt_ in enumerate(meshpts):
            # check direction
            meshpt = MatrixUtils.to2d(np.array(meshpt_))
            mesh_vec = meshpt - edge_midpt
            costheta = np.dot(MatrixUtils.unitize(edge_norm), MatrixUtils.unitize(mesh_vec))

            if costheta > 0.0:
                # is iwthin dist
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

                # TODO: theta not working! Figure out why
                # theta = MatrixUtils.angle(mesh_vec, edge_x_vector)

                dist_x = costheta  * np.linalg.norm(mesh_vec)
                within_x_dist = dist_x < max_x_dist
                within_y_dist = dist_y <= 3.0

                if within_x_dist and within_y_dist:
                    # move meshpt_ along normal back by h
                    proj_vec = -dist_y * norms[ei]
                    proj_pt = proj_vec + np.array(meshpt_)
                    close_pts[ei].append(proj_pt)
                    #ghclose.append(proj_pt.tolist())

    # sort pts by distance
    projected_points = []#close_pts#[None] * len(reduce(lambda a,b: a+b, close_pts))
    #ppl(close_pts)
    ghclose = []
    for ei, edgepts in enumerate(close_pts):
        refpt = edges[ei][0]
        flat_dist = lambda p: np.linalg.norm(MatrixUtils.to2d(p) - MatrixUtils.to2d(refpt))
        sorted_by_dist = sorted(close_pts[ei], key = flat_dist)
        refht_1 = sorted_by_dist[0].tolist()[2]
        refht_2 = sorted_by_dist[-1].tolist()[2]

        newedge1 = np.array([edges[ei][0][0], edges[ei][0][1], refht_1])
        projected_points.append(newedge1)
        projected_points.extend(sorted_by_dist)

    ghclose = [p.tolist() for p in projected_points]


    # weighted length average method
    grade_x_array, grade_y_array = np.zeros(len(projected_points)-1), np.zeros(len(projected_points)-1)

    for pi in range(len(projected_points) - 1):
        v1, v2 = projected_points[pi], projected_points[pi + 1]
        slope_vector = (v2 - v1)

        edge_vector  = (MatrixUtils.to2d(v2) - MatrixUtils.to2d(v1))
        #angle = MatrixUtil.angle(slope_vector, edge_vector)
        crossprod = np.cross(MatrixUtils.unitize(slope_vector), MatrixUtils.unitize(edge_vector))
        sintheta = np.linalg.norm(crossprod)
        grade_y = sintheta * np.linalg.norm(slope_vector)
        grade_x = np.linalg.norm(edge_vector)

        grade_x_array[pi] = grade_x
        grade_y_array[pi] = grade_y

    numerator = np.dot(grade_x_array, grade_y_array)
    denominator = np.dot(grade_y_array, np.ones(len(projected_points)-1))

    weighted_avg = 47.0#float(numerator/denominator)
    #print(weighted_avg)
    # Send to GH
    midpts = MatrixUtils.makelists(midpts)
    norms = MatrixUtils.makelists(norms)
    closepts = ghclose
    D = {}
    D["norm"] = norms
    D["mpts"] = midpts
    D["closepts"] = closepts
    D["number"] = weighted_avg
    interface.send(D)

    return projected_points

if __name__ == "__main__":

    # interface
    interface = Cache()
    interface.set_listeners()
    data_lst = project_points_to_line(interface)

    print("Success!", type(data_lst))
