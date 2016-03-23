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

def compute_angle(Coord, Coord_turb, directions = None):
    dim_points = Coord.shape[0]
    dim_turbines = Coord_turb.shape[0]
    angular_coeffs = np.zeros([dim_points, dim_turbines])
    verso_turb_point = np.zeros([dim_points, dim_turbines])
    if directions is not None:
        step = 180./directions
        directions_vect = np.arange(-90,90.1,step)
    for i,point in enumerate(Coord):
        point_xi = point[0]
        point_yi = point[1]
        current_angles = np.arctan2(-point_yi+Coord_turb[:,1],-point_xi+Coord_turb[:,0])
        current_angles_degree = np.degrees(current_angles)
        current_angles_degree = map_angle(current_angles_degree)
        if directions is not None:
            map_vect = np.argmin(np.abs(current_angles_degree.reshape([dim_turbines,1])-directions_vect.reshape(1,len(directions_vect))), axis = 1)
            current_angles_degree = directions_vect[map_vect]
        assert(np.logical_and(-90 <= np.all(current_angles_degree),np.all(current_angles_degree <=90)))

        angular_coeffs[i,:]= current_angles_degree
        verso = compute_verso_turbina_punto(Coord_turb, point_xi, point_yi)
        verso_turb_point[i,:] = verso

    return angular_coeffs, verso_turb_point

def compute_verso_turbina_punto(Coord_turb, point_xi, point_yi):
    diff_versus_x = np.sign(Coord_turb[:,0]-point_xi).reshape(Coord_turb.shape[0],1)
    diff_versus_y = np.sign(Coord_turb[:,1]-point_yi).reshape(Coord_turb.shape[0],1)

    verso = compute_final_verso(diff_versus_x,diff_versus_y)
    return verso

def compute_final_verso(verso_u,verso_v):
    verso = np.ones(len(verso_v))
    position_compared = np.any(verso_u == verso_v, axis=1)

    verso[np.intersect1d(np.where(position_compared==True)[0],np.where(verso_u == -1)[0])] = -1
    verso[np.intersect1d(np.where(position_compared==False)[0],np.where(verso_u == -1)[0])] = -0.5
    verso[np.intersect1d(np.where(position_compared==False)[0],np.where(verso_u == 1)[0])] = 0.5

    return verso

def compute_verso(u, v):
    verso_u = np.sign(u).reshape(len(u),1)
    verso_v = np.sign(v).reshape(len(v),1)
    verso = compute_final_verso(verso_u, verso_v)
    return verso

def invert_direction(verso_dir):
    verso_dir_copy = verso_dir.copy()
    verso_dir[verso_dir_copy==-1] = 1
    verso_dir[verso_dir_copy==1] = -1
    return verso_dir

def create_dict_direction(direction_train,directions):
    output_dict = {}
    n,p = direction_train.shape
    turbines_number = directions.shape[1]
    count = 0
    for i in range(p):
        current_angles = np.mean(direction_train[:,i])
        point_direction = np.array(directions[count,:]).reshape([turbines_number,1])
        medium_norm = np.linalg.norm(current_angles-point_direction, ord = 1,axis = 1)
        current_level = i%12
        if current_level==0 and i!=0:
                count+=1
        selected_turbs = np.where(medium_norm == medium_norm.min())[0]
        for selected_turb in selected_turbs:
            current_vect = np.array([count,current_level]).reshape([1,2])
            if selected_turb not in output_dict:
                output_dict[selected_turb] = current_vect
            else:
                output_dict[selected_turb] = np.concatenate((output_dict[selected_turb], current_vect), axis = 0)
    return output_dict



def map_angle(angle_to_map):
    extracted_angles = angle_to_map[np.logical_and(90 < angle_to_map,angle_to_map <=180)]
    if len(extracted_angles)!=0:
        extracted_angles = extracted_angles-180
    extracted_angles_neg = angle_to_map[np.logical_and(-180 < angle_to_map, angle_to_map <= -90)]
    if len(extracted_angles_neg)!=0:
        extracted_angles_neg = extracted_angles_neg+180
        angle_to_map[np.logical_and(-180 < angle_to_map, angle_to_map <= -90)] = extracted_angles_neg
    if len(extracted_angles)!=0:
        angle_to_map[np.logical_and(90 < angle_to_map,angle_to_map <=180)] = extracted_angles
    return angle_to_map
