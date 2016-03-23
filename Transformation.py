from abc import ABCMeta
import abc
import math
import numpy as np
import pandas as pd

from Enel_utils import create_tree, compute_verso, map_angle


class Transformation:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """fit poly"""

    def enel_transf_power_curve_multiple_key(self, keys, mean_value, power_curve):
        powers = []
        for k, key in enumerate(keys):
            if key != -1:
                values = power_curve[:, key * 2:key * 2 + 2]
                m = mean_value[k]
                mean_values_rounded = int(m) + 0.5
                row_power = np.where(values[:, 0] == mean_values_rounded)[0]
                if len(row_power) != 0:
                    row_power = row_power[0]
                    power = values[row_power, 1]
                else:
                    power = 0
            else:
                power = 0
            powers.append(power)
        return np.array(powers)

    def enel_transf_power_curve(self, key, mean_value, power_curve):
        values = power_curve[:, key * 2:key * 2 + 2]
        powers = []
        for m in mean_value:
            mean_values_rounded = int(m) + 0.5
            row_power = np.where(values[:, 0] == mean_values_rounded)[0]
            if len(row_power) != 0:
                row_power = row_power[0]
                power = values[row_power, 1]
            else:
                power = 0
            powers.append(power)
        return np.array(powers)

class Enel_conversionPowerCurve(Transformation):
    def __init__(self):
        pass

    def transform(self, XTrain, power_curve, Coord, Coord_turb, x_verso, dir_turbine,
                  x_verso_turbine):
        XTrain
        X_turbines = np.zeros([n, p])
        count = 0
        dict_turbs = dict.fromkeys(np.arange(0, turbines_number), np.array([[]], dtype="int64"))

        for i in range(p):
            selected_turbs = []
            current_level = i % 12
            if current_level == 0 and i != 0:
                count += 1
            current_angles = direction_train[:, i].reshape([n, 1])
            x_verso_current = x_verso[:, i]
            x_verso_turbine_current = x_verso_turbine[count, :].reshape(turbines_number, 1)
            point_direction = np.array(directions[count, :]).reshape([1, turbines_number])
            current_diff = np.abs(current_angles - point_direction)
            # min_turbs = np.min(current_diff,axis = 1)
            selected_turb_same_verso = np.any(x_verso_turbine_current == dir_turbine, axis=1)
            if np.any(selected_turb_same_verso):
                turb_same_verso = np.where(selected_turb_same_verso)[0]
                if len(turb_same_verso) != 0:
                    for s in range(n):
                        if x_verso_current[s] == dir_turbine:
                            min_turbs = np.min(current_diff[s, turb_same_verso])
                            turb = np.intersect1d(turb_same_verso, np.where(current_diff[s, :] == min_turbs)[0])

                            if len(turb) > 1:
                                # chooose neareast turbine in that direction
                                tree = create_tree(Coord_turb[turb, :])
                                d, sel_turb = tree.query(Coord[count, :])
                                selected_turb = turb[sel_turb]
                                key = selected_turb
                            else:
                                selected_turb = turb
                                key = selected_turb[0]
                        else:
                            # print("overcome trehold angle")
                            selected_turb = -1
                            # current_dict_values = np.array([count,current_level,s]).reshape([1,3])
                            # if dict_turbs[key].shape[1]==0:
                            #   dict_turbs[key] = current_dict_values
                            # else:
                            #   dict_turbs[key] = np.concatenate((dict_turbs[key],current_dict_values), axis = 0)
                        selected_turbs.append(selected_turb)
                    wind_speed = XTrain[:, i]
                    power_values = self.enel_transf_power_curve_multiple_key(selected_turbs, wind_speed, power_curve)

                    X_turbines[:, i] = power_values
        return X_turbines, dict_turbs

class Enel_directionVersoPowerCurveTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, direction_train, directions, XTrain, power_curve, Coord, Coord_turb, x_verso, dir_turbine,
                  x_verso_turbine, compute_dict):
        n, p = direction_train.shape
        turbines_number = directions.shape[1]
        X_turbines = np.zeros([n, p])
        count = 0
        dict_turbs = dict.fromkeys(np.arange(0, turbines_number), np.array([[]], dtype="int64"))

        for i in range(p):
            selected_turbs = []
            current_level = i % 12
            if current_level == 0 and i != 0:
                count += 1
            current_angles = direction_train[:, i].reshape([n, 1])
            x_verso_current = x_verso[:, i]
            x_verso_turbine_current = x_verso_turbine[count, :].reshape(turbines_number, 1)
            point_direction = np.array(directions[count, :]).reshape([1, turbines_number])
            current_diff = np.abs(current_angles - point_direction)
            # min_turbs = np.min(current_diff,axis = 1)
            selected_turb_same_verso = np.any(x_verso_turbine_current == dir_turbine, axis=1)
            if np.any(selected_turb_same_verso):
                turb_same_verso = np.where(selected_turb_same_verso)[0]
                if len(turb_same_verso) != 0:
                    for s in range(n):
                        if x_verso_current[s] == dir_turbine:
                            min_turbs = np.min(current_diff[s, turb_same_verso])
                            turb = np.intersect1d(turb_same_verso, np.where(current_diff[s, :] == min_turbs)[0])

                            if len(turb) > 1:
                                # chooose neareast turbine in that direction
                                tree = create_tree(Coord_turb[turb, :])
                                d, sel_turb = tree.query(Coord[count, :])
                                selected_turb = turb[sel_turb]
                                key = selected_turb
                            else:
                                selected_turb = turb
                                key = selected_turb[0]
                        else:
                            # print("overcome trehold angle")
                            selected_turb = -1
                            if compute_dict:
                                current_dict_values = np.array([count,current_level,s]).reshape([1,3])
                                if dict_turbs[key].shape[1]==0:
                                  dict_turbs[key] = current_dict_values
                                else:
                                  dict_turbs[key] = np.concatenate((dict_turbs[key],current_dict_values), axis = 0)
                        selected_turbs.append(selected_turb)
                    wind_speed = XTrain[:, i]
                    power_values = self.enel_transf_power_curve_multiple_key(selected_turbs, wind_speed, power_curve)

                    X_turbines[:, i] = power_values
        return X_turbines, dict_turbs

    def transform_verso_distance(self, XTrain, power_curve, Coord, Coord_turb, x_verso, dir_turbine, x_verso_turbine):
        n, p = XTrain.shape
        turbines_number = Coord_turb.shape[0]
        X_turbines = np.zeros([n, p])
        count = 0
        dict_turbs = dict.fromkeys(np.arange(0, turbines_number), np.array([[]], dtype="int64"))

        for i in range(p):
            selected_turbs = []
            current_level = i % 12
            if current_level == 0 and i != 0:
                count += 1
            x_verso_current = x_verso[:, i]
            x_verso_turbine_current = x_verso_turbine[count, :].reshape(turbines_number, 1)

            # min_turbs = np.min(current_diff,axis = 1)
            selected_turb_same_verso = np.any(x_verso_turbine_current == dir_turbine, axis=1)
            if np.any(selected_turb_same_verso):
                turb_same_verso = np.where(selected_turb_same_verso)[0]
                if len(turb_same_verso) != 0:
                    for s in range(n):
                        if x_verso_current[s] == dir_turbine:
                            # chooose neareast turbine in that direction
                            tree = create_tree(Coord_turb)
                            d, selected_turb = tree.query(Coord[count, :], k=1)
                            key = selected_turb
                        else:
                            # print("overcome trehold angle")
                            selected_turb = -1
                            # current_dict_values = np.array([count,current_level,s]).reshape([1,3])
                            # if dict_turbs[key].shape[1]==0:
                            #   dict_turbs[key] = current_dict_values
                            # else:
                            #   dict_turbs[key] = np.concatenate((dict_turbs[key],current_dict_values), axis = 0)
                        selected_turbs.append(selected_turb)
                    wind_speed = XTrain[:, i]
                    power_values = self.enel_transf_power_curve(selected_turbs, wind_speed, power_curve)

                    X_turbines[:, i] = power_values
        return X_turbines, dict_turbs


class Enel_directionPowerCurveTransformation(Transformation):

    def transform_versus(self, direction_train, directions, X_speed, power_curve, Coord, Coord_turb, x_verso, x_verso_turbine,threshold_dir = 180, compute_dict=1):
        n,p = direction_train.shape
        turbines_number = directions.shape[1]
        X_turbines = np.zeros([n,p])
        count = 0
        dict_turbs = dict.fromkeys(np.arange(0,turbines_number),np.array([[]], dtype = "int64"))
        if compute_dict:
            current_dict_values = np.zeros([n,3])
            current_dict_values[:,0] = count
            current_dict_values[:,2] = np.arange(0,n)
        for i in range(p):
            selected_turbs = []
            keys = []
            current_level = i%12
            if compute_dict:
                current_dict_values[:,1] = current_level
            if current_level==0 and i!=0:
                count+=1
                print("punto", count)
                if compute_dict:
                    current_dict_values[:,0] = count
            current_angles = direction_train[:,i].reshape([n,1])
            point_direction = np.array(directions[count,:]).reshape([1,turbines_number])
            current_diff = np.abs(current_angles-point_direction)
            x_verso_current = x_verso[:, i]
            x_verso_turbine_current = x_verso_turbine[count, :].reshape(turbines_number, 1)
            for s in range(n):
                x_verso_current_sample = x_verso_current[s]
                selected_turb_same_verso = np.any(x_verso_turbine_current == x_verso_current_sample, axis=1)
                if np.any(selected_turb_same_verso):
                    turb_same_verso = np.where(selected_turb_same_verso)[0]
                    min_turbs = np.min(current_diff[s, turb_same_verso])
                    if min_turbs<threshold_dir:
                        turb = np.intersect1d(turb_same_verso, np.where(current_diff[s, :] == min_turbs)[0])
                        if len(turb) > 1:
                        # chooose neareast turbine in that direction
                            tree = create_tree(Coord_turb[turb, :])
                            d, sel_turb = tree.query(Coord[count, :])
                            selected_turb = turb[sel_turb]
                            if compute_dict:
                                key = selected_turb
                        else:
                            selected_turb = turb
                            if compute_dict:
                                key = selected_turb[0]
                        if compute_dict:
                            keys.append(key)
                    else:
                        selected_turb = -1
                else:
                    selected_turb = -1
                selected_turbs.append(selected_turb)
            if compute_dict:
                for turb_key in np.unique(keys):
                    position_turb = np.where(keys==turb_key)[0]
                    if dict_turbs[turb_key].shape[1]==0:
                        dict_turbs[turb_key] = current_dict_values[position_turb,:]
                    else:
                        dict_turbs[turb_key] = np.concatenate((dict_turbs[turb_key],current_dict_values[position_turb,:]), axis = 0)

            wind_speed = X_speed[:, i]
            power_values = self.enel_transf_power_curve_multiple_key(selected_turbs, wind_speed, power_curve)
            X_turbines[:,i] = power_values
        return X_turbines,dict_turbs

    def transform(self, direction_train, directions, X_speed, power_curve, Coord, Coord_turb, threshold_dir=180,compute_dict=1):
        n, p = direction_train.shape
        turbines_number = directions.shape[1]
        X_turbines = np.zeros([n, p])
        count = 0
        dict_turbs = dict.fromkeys(np.arange(0, turbines_number), np.array([[]], dtype="int64"))
        if compute_dict:
            current_dict_values = np.zeros([n, 3])
            current_dict_values[:, 0] = count
            current_dict_values[:, 2] = np.arange(0, n)
        for i in range(p):
            selected_turbs = []
            keys = []
            current_level = i % 12
            if compute_dict:
                current_dict_values[:, 1] = current_level
            if current_level == 0 and i != 0:
                count += 1
                print("punto", count)
                if compute_dict:
                    current_dict_values[:, 0] = count
            current_angles = direction_train[:, i].reshape([n, 1])
            point_direction = np.array(directions[count, :]).reshape([1, turbines_number])
            current_diff = np.abs(current_angles - point_direction)
            min_turbs = np.min(current_diff, axis=1)

            for s in range(n):
                if min_turbs[s] < threshold_dir:
                    turb = np.where(current_diff[s, :] == min_turbs[s])[0]
                    if len(turb) > 1:
                        # chooose neareast turbine in that direction
                        tree = create_tree(Coord_turb[turb, :])
                        d, sel_turb = tree.query(Coord[count, :])
                        selected_turb = turb[sel_turb]
                        if compute_dict:
                            key = selected_turb
                    else:
                        selected_turb = turb
                        if compute_dict:
                            key = selected_turb[0]
                    if compute_dict:
                        keys.append(key)
                else:
                    selected_turb = -1
                selected_turbs.append(selected_turb)
            if compute_dict:
                for turb_key in np.unique(keys):
                    position_turb = np.where(keys == turb_key)[0]
                    if dict_turbs[turb_key].shape[1] == 0:
                        dict_turbs[turb_key] = current_dict_values[position_turb, :]
                    else:
                        dict_turbs[turb_key] = np.concatenate(
                            (dict_turbs[turb_key], current_dict_values[position_turb, :]), axis=0)

            wind_speed = X_speed[:, i]
            power_values = self.enel_transf_power_curve_multiple_key(selected_turbs, wind_speed, power_curve)
            X_turbines[:, i] = power_values
        return X_turbines, dict_turbs


    def transformPerTurbineLevel(self, neigh_, dict_, x, power_curve, x_transf, output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]
            if current_value.shape[1] != 0:
                current_point_level = current_value
                current_features = current_point_level[:, 0:2]
                current_samples = current_point_level[:, 2]
                b = np.ascontiguousarray(current_features).view(
                    np.dtype((np.void, current_features.dtype.itemsize * current_features.shape[1])))
                _, idx = np.unique(b, return_index=True)

                unique_feat_point_level = current_features[idx]
                x_tmp_u = np.zeros([n, unique_feat_point_level.shape[0]])
                x_tmp_v = np.zeros([n, unique_feat_point_level.shape[0]])

                index_to_sum = {}
                for j, f in enumerate(unique_feat_point_level):
                    index_f = np.where(np.all(current_features == f, axis=1))[0]
                    index_sample = current_samples[index_f]
                    absent_s = np.delete(np.arange(n), index_sample)
                    column_u = dict_[f[0]][f[1]]
                    x_tmp_u[:, j] = x[:, column_u]
                    x_tmp_u[absent_s, j] = 0
                    column_v = dict_[f[0]][f[1] + 12]
                    x_tmp_v[:, j] = x[:, column_v]
                    x_tmp_v[absent_s, j] = 0
                    if f[1] not in index_to_sum:
                        index_to_sum[f[1]] = []
                    index_to_sum[f[1]].append(j)
                for level in index_to_sum:
                    equal_levels = index_to_sum[level]
                    sum_component_u = np.sum(x_tmp_u[:,equal_levels], axis = 1)
                    sum_component_v = np.sum(x_tmp_v[:,equal_levels], axis = 1)
                    wind_speed = np.sqrt(sum_component_u**2+sum_component_v**2)/len(equal_levels)
                    power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                    current_dim = x_transf.shape[1]
                    if current_dim==0:
                        x_transf = power_value.reshape([n,1])
                    else:
                        x_transf = np.concatenate((x_transf,power_value.reshape([n,1])), axis = 1)
                    for feat in unique_feat_point_level[equal_levels,0]:
                        vect_to_append = np.array([current_dim,level]).reshape(1,2)
                        if output_dict_[feat].shape[1]==0:
                            output_dict_[feat] = vect_to_append
                        else:
                            output_dict_[feat] = np.concatenate((output_dict_[feat], vect_to_append),axis = 0)
        return x_transf, output_dict_


    def transformPerTurbine(self, neigh_, dict_, x, power_curve, x_transf, output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]
            if current_value.shape[1] != 0:
                current_point_level = current_value
                current_features = current_point_level[:, 0:2]
                current_samples = current_point_level[:, 2]
                b = np.ascontiguousarray(current_features).view(
                    np.dtype((np.void, current_features.dtype.itemsize * current_features.shape[1])))
                _, idx = np.unique(b, return_index=True)

                unique_feat_point_level = current_features[idx]
                x_tmp_u = np.zeros([n, unique_feat_point_level.shape[0]])
                x_tmp_v = np.zeros([n, unique_feat_point_level.shape[0]])
                start_dim = x_transf.shape[1]
                for j, f in enumerate(unique_feat_point_level):
                    index_f = np.where(np.all(current_features == f, axis=1))[0]
                    index_sample = current_samples[index_f]
                    absent_s = np.delete(np.arange(n), index_sample)
                    column_u = dict_[f[0]][f[1]]
                    x_tmp_u[:, j] = x[:, column_u]
                    x_tmp_u[absent_s, j] = 0
                    column_v = dict_[f[0]][f[1] + 12]
                    x_tmp_v[:, j] = x[:, column_v]
                    x_tmp_v[absent_s, j] = 0
                sum_component_u = np.sum(x_tmp_u, axis=1)
                sum_component_v = np.sum(x_tmp_v, axis=1)
                wind_speed = np.sqrt(sum_component_u ** 2 + sum_component_v ** 2) / len(np.unique(current_features))
                power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                if x_transf.shape[1] == 0:
                    x_transf = power_value.reshape([n, 1])
                else:
                    x_transf = np.concatenate((x_transf, power_value.reshape([n, 1])), axis=1)
                current_dim = x_transf.shape[1]
                unique_feat_point = np.unique(unique_feat_point_level[:, 0])
                for current_v in unique_feat_point:
                    current_index_levels = np.where(unique_feat_point_level[:, 0] == current_v)[0]
                    current_levels = np.sort(unique_feat_point_level[current_index_levels, 1])
                    indexes_col = np.arange(start_dim, current_dim)
                    vect_to_append = np.zeros([len(indexes_col) * len(current_levels), 2])
                    vect_to_append[:, 0] = indexes_col
                    vect_to_append[:, 1] = current_levels
                    if output_dict_[current_v].shape[1] == 0:
                        output_dict_[current_v] = vect_to_append
                    else:
                        output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append), axis=0)
        return x_transf, output_dict_


class Enel_powerCurveTransformation(Transformation):
    def __init__(self):
        pass

    def compute_angle_matrix(self,x, directions = None):
        n,m = x.shape
        x_transf = np.array([[]])
        x_verso = np.array([[]])
        dict_ = dict.fromkeys(np.arange(0,49),np.array([]))
        key = 0
        if directions is not None:
            step = 180./directions
            directions_vect = np.arange(-90,90.1,step)
        for i in range(0,m,24):
            start_dim = x_transf.shape[1]
            for j in range(i,i+12):
                current_angle = np.arctan2(x[:,j+12],x[:,j])
                current_angle_degree = np.degrees(current_angle)
                current_angle_degree = map_angle(current_angle_degree).reshape([n,1])
                if directions is not None:
                    map_vect = np.argmin(np.abs(current_angle_degree-directions_vect.reshape(1,len(directions_vect))), axis = 1)
                    current_angle_degree = directions_vect[map_vect].reshape([n,1])

                verso_current = compute_verso(x[:, j], x[:, j + 12])
                verso_current = verso_current.reshape(len(verso_current),1)
                if x_transf.shape[1]==0:
                    x_transf = current_angle_degree
                    x_verso = verso_current
                else:
                    x_transf = np.concatenate((x_transf,current_angle_degree), axis = 1)
                    x_verso = np.concatenate((x_verso,verso_current), axis = 1)

            current_dim = x_transf.shape[1]
            dict_[key] = np.append(dict_[key], np.arange(start_dim,current_dim))
            key+=1

        assert (x_transf.shape[1]==m/2)
        return x_transf, x_verso, dict_

    def transform(self, neigh_, dict_, x, power_curve, l, x_transf, output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        k_levels = np.arange(0, 12)
        for key in keys_:
            print(key)
            current_value = values[key]
            if len(current_value) != 0:
                if l == 0:
                    h_s = np.arange(0, len(current_value))
                else:
                    h_s = np.arange(len(current_value) - 1, len(current_value))
                for h in h_s:
                    if l == 0:
                        current_point_level = current_value[h]
                        current_values = np.array([current_point_level])
                    else:
                        current_point_level = current_value[:h + 1]
                        current_values = current_point_level
                    start_dim = x_transf.shape[1]
                    for k in k_levels:
                        sum_component_u = self.get_component_value_old(x, dict_, k, current_values)
                        sum_component_v = self.get_component_value_old(x, dict_, k + 12, current_values)
                        wind_speed = np.sqrt(sum_component_u ** 2 + sum_component_v ** 2) / len(current_values)
                        power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                        if x_transf.shape[1] == 0:
                            x_transf = power_value.reshape([n, 1])
                        else:
                            x_transf = np.concatenate((x_transf, power_value.reshape([n, 1])), axis=1)
                current_dim = x_transf.shape[1]
                for i, current_v in enumerate(current_values):
                    vect_to_append = np.array([np.arange(start_dim, current_dim)[0], k_levels[i]]).reshape([1, 2])
                    if output_dict_[current_v].shape[1] == 0:
                        output_dict_[current_v] = vect_to_append
                    else:
                        output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append), axis=0)
        return x_transf, output_dict_

    def transform1(self, neigh_, dict_, x, power_curve, l, x_transf, output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for index_key, key in enumerate(keys_):
            current_value = values[index_key]
            if len(current_value) != 0:
                if l == 0:
                    h_s = np.arange(0, len(current_value[:, 0]))
                else:
                    h_s = np.arange(len(current_value[:, 0]) - 1, len(current_value[:, 0]))
                for h in h_s:
                    if l == 0:
                        current_point_level = current_value[h]
                        current_values = np.array([current_point_level])[:, 0]
                        k_levels = np.array([current_point_level])[:, 1]
                    else:
                        current_point_level = current_value[:h + 1]
                        current_values = current_point_level[:, 0]
                        k_levels = current_point_level[:, 1]
                    start_dim = x_transf.shape[1]

                    for k in np.unique(k_levels):
                        index_k = np.where(k_levels == k)[0]
                        values_k = current_values[index_k]

                        sum_component_u = self.get_component_value_old(x, dict_, k, values_k)
                        sum_component_v = self.get_component_value_old(x, dict_, k + 12, values_k)
                        wind_speed = np.sqrt(sum_component_u ** 2 + sum_component_v ** 2) / len(values_k)
                        power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                        if x_transf.shape[1] == 0:
                            x_transf = power_value.reshape([n, 1])
                        else:
                            x_transf = np.concatenate((x_transf, power_value.reshape([n, 1])), axis=1)
                        current_dim = x_transf.shape[1]
                    for i, current_v in enumerate(current_values):
                        vect_to_append = np.array([np.arange(start_dim, current_dim)[0], k_levels[i]]).reshape([1, 2])
                        if output_dict_[current_v].shape[1] == 0:
                            output_dict_[current_v] = vect_to_append
                        else:
                            output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append), axis=0)
        return x_transf, output_dict_

    def get_component_value_old(self, x, dict_, k, current_values):
        c = [dict_[j].astype("int64")[k] for j in current_values]
        c = np.hstack(c).astype("int64")
        current_vect = x[:, c]
        sum_component = np.sum(current_vect, axis=1)
        return sum_component

    def get_component_value(self, x, dict_, k, current_values):
        c = [dict_[j].astype("int64")[k[index]] for index, j in enumerate(current_values)]
        c = np.hstack(c).astype("int64")
        current_vect = x[:, c]
        sum_component = np.sum(current_vect, axis=1)
        return sum_component



class EnelWindSpeedTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, x):
        n, m = x.shape
        x_transf = np.array([[]])
        dict_ = dict.fromkeys(np.arange(0, 49), np.array([]))
        wind_direction = []
        key = 0
        for i in range(0, m, 24):
            start_dim = x_transf.shape[1]
            for j in range(i, i + 12):
                wind_speed = np.sqrt(np.power(x[:, j], 2) + np.power(x[:, j + 12], 2)).reshape([n, 1])
                if x_transf.shape[1] == 0:
                    x_transf = wind_speed
                else:
                    x_transf = np.concatenate((x_transf, wind_speed), axis=1)
            current_u = np.sum(x[:, :j], axis = 1)
            current_v = np.sum(x[:, :j+12], axis = 1)
            current_dir = np.degrees(np.arctan2(current_v, current_u))
            wind_direction.append(current_dir)
            current_dim = x_transf.shape[1]
            dict_[key] = np.append(dict_[key], np.arange(start_dim, current_dim))
            key += 1

        assert (x_transf.shape[1] == m / 2)

        return x_transf, dict_, wind_direction


class inverseTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, x):
        x_transf = 1. / x
        return x_transf


class cosTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, x):
        x_transf = np.cos(x)
        return x_transf


class sinTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, x):
        x_transf = np.sin(x)
        return x_transf


class logTrasformation(Transformation):
    def __init__(self):
        pass

    def transform(self, x):
        x_transf = np.log(x)
        return x_transf


class expTransformation(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self, x):
        n, m = x.shape
        x_transf = np.zeros([n, m * (self.degree)])
        for d in range(self.degree):
            x_transf[:, m * (d):m * (d) + m] = np.exp(x ** (d + 1))
        return x_transf


class NullTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, X):
        # onesVec = np.ones([len(X), 1])
        # X = np.append(onesVec, X, axis=1)
        return X


class F1(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1])

    def transform(self, x):
        y = (x[0] ** 4 - x[0] ** 2) * (3 + x[1])
        return y, self.informative  # [-3,3]

    def expand(self, x):
        new_informatives = np.array([])
        n, m = x.shape
        dict_ = dict.fromkeys(np.arange(0, m), np.array([]))
        keys_ = np.array(dict_.keys())
        powers = [2, 4]
        len_powers = len(powers)
        dim = len(powers) * m + len(powers) * m * m
        x_transf = np.zeros([n, dim])
        for d in range(len_powers):
            x_transf[:, m * (d):m * (d) + m] = np.power(x, powers[d])
            new_informatives = np.append(new_informatives, self.informative + m * d)
            for key in keys_:
                dict_[key] = np.append(dict_[key], key + m * d)
        for col in range((len_powers) * m):  ##x^2*x, x^4*x
            x_transf[:, len_powers * m + col * m:len_powers * m + col * m + m] = x_transf[:, col].reshape([n, 1]) * x
            if col in new_informatives:
                new_informatives = np.append(new_informatives,
                                             np.arange(len_powers * m + col * m, len_powers * m + col * m + 2))
            right_feat = col % (m)
            dict_[right_feat] = np.append(dict_[right_feat],
                                          np.arange(len_powers * m + col * m, len_powers * m + col * m + m))
            for key in keys_:
                if key != right_feat:
                    dict_[key] = np.append(dict_[key], len_powers * m + col * m + key)

        return x_transf, new_informatives, dict_  # [-3,3]


class F2(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1])

    def transform(self, x):
        y = 2 * (x[0] ** 3 - x[0]) * (2 * x[1] - 1) * (x[1] + 1) + (x[1] ** 3 - x[1] + 3)
        return y, self.informative  # [-3,3]

    def expand(self, x):
        new_informatives = np.array([])
        n, m = x.shape
        dict_ = dict.fromkeys(np.arange(0, m), np.array([]))
        keys_ = np.array(dict_.keys())
        powers = [1, 3]
        len_powers = len(powers)

        dim = m * 2
        x_transf = np.zeros([n, dim])
        for d in range(len_powers):  ##x e x^3
            x_transf[:, m * (d):m * (d) + m] = np.power(x, powers[d])
            new_informatives = np.append(new_informatives, self.informative + m * d)
            for key in keys_:
                dict_[key] = np.append(dict_[key], key + m * d).astype("int64")

        for count in range(m - 1):
            current_dim = x_transf.shape[1]
            x_transf = np.concatenate((x_transf, x[:, count].reshape([n, 1]) * x[:, count + 1:]), axis=1)
            new_dim = x_transf.shape[1]
            diff_dim = new_dim - current_dim
            if count in new_informatives:
                for a in range(0, diff_dim):
                    if count + a + 1 in new_informatives:
                        new_informatives = np.append(new_informatives, current_dim + a)
            dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
            for key in keys_:
                if key > count:
                    dict_[key] = np.append(dict_[key], current_dim + key - count - 1)

        for count in range(m):
            for count1 in range(m):
                if count != count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate(
                        (x_transf, x[:, count].reshape([n, 1]) * np.power(x[:, count1].reshape([n, 1]), 2)), axis=1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                        new_informatives = np.append(new_informatives, np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        for count in range(m):
            for count1 in range(m):
                if count != count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate(
                        (x_transf, x[:, count].reshape([n, 1]) * np.power(x[:, count1].reshape([n, 1]), 3)), axis=1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                        new_informatives = np.append(new_informatives, np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        for count in range(m):
            for count1 in range(m):
                if count != count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate((x_transf,
                                               np.power(x[:, count].reshape([n, 1]), 2).reshape([n, 1]) * np.power(
                                                   x[:, count1].reshape([n, 1]), 3)), axis=1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                        new_informatives = np.append(new_informatives, np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        assert (x_transf.shape[1] == m * 2 + m * (m - 1) / 2 + m * (m - 1) * 3)
        return x_transf, new_informatives, dict_  # [-3,3]


class F3(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1])

    def transform(self, x):
        y = -2 * (2 * x[0] ** 2 - 1) * x[1] * math.exp(-x[0] ** 2 - x[1] ** 2)
        return y, self.informative  # [-1,1]

    def expand(self, x):
        new_informatives = np.array([])
        n, m = x.shape
        x_transf = np.array([])
        dict_ = dict.fromkeys(np.arange(0, m), np.array([]))
        keys_ = np.array(dict_.keys())

        for count1 in range(m):
            for count2 in range(m):  ##x*exp(-x-x)
                if count2 > count1:
                    if len(x_transf) != 0:
                        current_dim = x_transf.shape[1]
                    else:
                        current_dim = 0
                    a = x * np.exp(-np.power(x[:, count1], 2) - np.power(x[:, count2], 2)).reshape([n, 1])
                    if count1 == 0 and count2 == 1:
                        x_transf = a
                    else:
                        x_transf = np.concatenate((x_transf, a), axis=1)

                    new_dim = x_transf.shape[1]
                    if count1 in self.informative:
                        if count2 in self.informative:
                            new_informatives = np.append(new_informatives,
                                                         np.arange(current_dim, current_dim + len(self.informative)))
                    if count1 != count2:
                        dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim)).astype("int64")
                    dict_[count2] = np.append(dict_[count2], np.arange(current_dim, new_dim)).astype("int64")

                    for key in keys_:
                        if key != count2:
                            if key != count1:
                                dict_[key] = np.append(dict_[key], current_dim + key).astype("int64")
        dim_x_transf = x_transf.shape[1]
        x_transf_copy = x_transf.copy()
        values_ = dict_.values()
        for count1 in range(m):
            current_dim = x_transf.shape[1]
            del_vect = np.arange(count1, dim_x_transf, step=m)

            x_transf_tmp = np.delete(x_transf_copy, del_vect, axis=1)

            x_transf = np.concatenate((x_transf, np.power(x[:, count1], 2).reshape([n, 1]) * x_transf_tmp), axis=1)
            new_dim = x_transf.shape[1]
            dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim)).astype("int64")
            for key in keys_:
                if key != count1:
                    new_values = np.array([]).astype("int64")
                    for v in values_[key]:
                        if v not in del_vect:
                            elem_less = len(del_vect[del_vect < v])
                            new_e = v - elem_less
                            new_values = np.append(new_values, new_e)
                    new_values = np.unique(new_values)
                    dict_[key] = np.append(dict_[key], new_values + current_dim).astype("int64")
            if count1 in self.informative:
                new_informatives = np.append(new_informatives, current_dim)
        return x_transf, new_informatives, dict_


class F4(Transformation):
    def __init__(self):
        self.informative = informative = np.array([0, 1, 2])

    def transform(self, x):
        y = x[0] + (x[1] > 0.5) * (x[2] > 0.5)
        return y, self.informative  # [0,1]

    def expand(self, x):
        new_informatives = self.informative.copy()
        n, m = x.shape
        dict_ = dict.fromkeys(np.arange(0, m), np.array([]))
        keys_ = np.array(dict_.keys())
        for key in keys_:
            dict_[key] = key
        x_transf = x.copy()
        for count1 in range(m):
            for count2 in range(m):
                current_dim = x_transf.shape[1]
                if count2 >= count1:
                    a = (x[:, count1] > 0.5).reshape([n, 1]) * (x[:, count2] > 0.5).reshape([n, 1])
                    x_transf = np.concatenate((x_transf, a), axis=1)
                    if count1 in self.informative and count2 in self.informative:
                        new_informatives = np.append(new_informatives, current_dim)
                    if count1 != count2:
                        dict_[count1] = np.append(dict_[count1], current_dim).astype("int64")
                    dict_[count2] = np.append(dict_[count2], current_dim).astype("int64")
        return x_transf, new_informatives, dict_  # [0,1]


class F5(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1, 2, 3, 4])

    def transform(self, x):
        y = 10 * math.sin(x[0]) * x[1] + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]
        return y, self.informative  # [0,1]

    def expand(self, x):
        new_informatives = self.informative.copy()
        n, m = x.shape
        x_transf = x.copy()
        x_transf = np.concatenate((x_transf, np.power(x, 2)), axis=1)
        dict_ = dict.fromkeys(np.arange(0, m), np.array([]))
        keys_ = np.array(dict_.keys())
        for key in keys_:
            dict_[key] = np.append(dict_[key], [key, key + m]).astype("int64")
        new_informatives = np.append(new_informatives, np.arange(m, m + len(self.informative)))

        for count1 in range(m):
            current_dim = x_transf.shape[1]
            a = x * np.sin(x[:, count1]).reshape([n, 1])
            x_transf = np.concatenate((x_transf, a), axis=1)
            if count1 in self.informative:
                new_informatives = np.append(new_informatives,
                                             np.arange(current_dim, current_dim + len(self.informative)))
            dict_[count1] = np.append(dict_[count1], np.arange(current_dim, current_dim + m))
            for key in keys_:
                if key != count1:
                    dict_[key] = np.append(dict_[key], key + current_dim).astype("int64")

        return x_transf, new_informatives, dict_  # [0,1]
