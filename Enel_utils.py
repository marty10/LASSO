import numpy as np
from scipy import spatial

def extract_new_dict(dict_):

    levels = np.arange(0,12)
    values = (list)(dict_.values())
    output_dict = {}
    for key, v in enumerate(values):
        if key not in output_dict:
            output_dict[key] = {}
        for level in levels:
            value_level = v[v[:,1]==level][:,0]
            if level not in output_dict[key]:
                output_dict[key][level] = np.array([], dtype = "int64")
            output_dict[key][level] = np.append(output_dict[key][level],value_level)
    return output_dict



def find_nearest(Coord,k):
    dim = Coord.shape[0]
    dict_ = dict.fromkeys(np.arange(0,dim),np.array([]))
    tree = create_tree(Coord)
    for i in range(0,dim):
        d,j = tree.query(Coord[i,:],k = k)
        j = j[j>=i]
        dict_[i] = j
    return dict_

def find_turbines_nearest_points(Coord,Coord_turb,k):
    dim = Coord.shape[0]
    dict_ = dict.fromkeys(np.arange(0,Coord_turb.shape[0]),np.array([]))
    tree = create_tree(Coord_turb)
    for i in range(0,dim):
        d,k_nearest = tree.query(Coord[i,:],k = k)
        if k==1:
            dict_[k_nearest] = np.append(dict_[k_nearest], i)
        else:
            for j in (k_nearest):
                dict_[j] = np.append(dict_[j], i)
    return dict_

def find_nearest_turbine(Coord,Coord_turb,k):
    dim = Coord_turb.shape[0]
    #dict_ = dict.fromkeys(np.arange(0,dim),np.array([k,2]))
    dict_ = {}
    tree = create_tree(Coord)
    for i in range(0,dim):
        d,j = tree.query(Coord_turb[i,:],k = k)
        if i not in dict_:
            dict_[i] = np.zeros([k,2])
        dict_[i][:,0] = j
        dict_[i][:,1] = d
    return dict_

def create_tree(points):
    x = points[:,0]
    y = points[:,1]
    tree = spatial.KDTree(np.column_stack((x.ravel(), y.ravel())))
    return tree

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
