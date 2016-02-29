import numpy as np
from scipy import spatial


def find_nearest(Coord,k):
    dim = Coord.shape[0]
    dict_ = dict.fromkeys(np.arange(0,dim),np.array([]))
    x = Coord[:,0]
    y = Coord[:,1]

    tree = spatial.KDTree(np.column_stack((x.ravel(), y.ravel())))
    for i in range(0,dim):
        d,j = tree.query(Coord[i,:],k = k)
        j = j[j>=i]
        dict_[i] = j
    return dict_


def find_nearest_turbine(Coord,Coord_turb,k):
    dim = Coord_turb.shape[0]
    #dict_ = dict.fromkeys(np.arange(0,dim),np.array([k,2]))
    dict_ = {}
    x = Coord[:,0]
    y = Coord[:,1]

    tree = spatial.KDTree(np.column_stack((x.ravel(), y.ravel())))
    for i in range(0,dim):
        d,j = tree.query(Coord_turb[i,:],k = k)
        if i not in dict_:
            dict_[i] = np.zeros([k,2])
        dict_[i][:,0] = j
        dict_[i][:,1] = d
    return dict_


def get_products_points(turb_dict):

    values_ = np.array((list)(turb_dict.values()))
    v_tot = np.unique(np.hstack(values_))
    output_dict = dict.fromkeys(np.arange(0,49),np.array([]))
    for key in output_dict.keys():
        output_dict[key] = np.array([key])

    for v in v_tot:
        v_row = np.where(np.any(values_==v, axis = 1))
        tmp_v = np.unique(np.hstack(values_[v_row]))
        output_dict[v] = np.append(output_dict[v],tmp_v[tmp_v>v])
    return output_dict
