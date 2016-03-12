from abc import ABCMeta
import abc
import math
import numpy as np
from Enel_utils import create_tree

class Transformation:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """fit poly"""


class Enel_directionPowerCurveTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self, direction_train, directions, XTrain, power_curve, Coord, Coord_turb):
        n,p = direction_train.shape
        turbines_number = directions.shape[1]
        X_turbines = np.zeros([n,p])
        count = 0
        #dict_turbs = dict.fromkeys(np.arange(0,turbines_number),np.array([[]], dtype = "int64"))

        for i in range(p):
            selected_turbs = []
            current_level = i%12
            if current_level==0 and i!=0:
                count+=1
            current_angles = direction_train[:,i].reshape([n,1])
            point_direction = np.array(directions[count,:]).reshape([1,turbines_number])
            current_diff = np.abs(current_angles-point_direction)
            min_turbs = np.min(current_diff,axis = 1)

            for s in range(n):
                if min_turbs[s]<180:
                    turb = np.where(current_diff[s,:] == min_turbs[s])[0]
                    if len(turb)>1:
                    #chooose neareast turbine in that direction
                        tree = create_tree(Coord_turb[turb,:])
                        d,sel_turb = tree.query(Coord[i%12,:])
                        selected_turb = turb[sel_turb]
                        key = selected_turb
                    else:
                        selected_turb = turb
                        key = selected_turb[0]
                else:
                    print("overcome trehold angle")
                # current_dict_values = np.array([count,current_level,s]).reshape([1,3])
                # if dict_turbs[key].shape[1]==0:
                #     dict_turbs[key] = current_dict_values
                # else:
                #     dict_turbs[key] = np.concatenate((dict_turbs[key],current_dict_values), axis = 0)
                selected_turbs.append(selected_turb)
            wind_speed = XTrain[:,i]
            power_values = self.enel_transf_power_curve(selected_turbs, wind_speed, power_curve)
            X_turbines[:,i] = power_values
        return X_turbines#,dict_turbs

    def enel_transf_power_curve(self, keys, mean_value, power_curve):
        powers = []
        for k,key in enumerate(keys):
            if key!=-1:
                values = power_curve[:,key*2:key*2+2]
                m = mean_value[k]
                mean_values_rounded= int(m)+0.5
                row_power = np.where(values[:,0]==mean_values_rounded)[0]
                if len(row_power)!=0:
                    row_power = row_power[0]
                    power = values[row_power,1]
                else:
                    power = 0
            else:
                power = 0
            powers.append(power)
        return np.array(powers)

    def enel_transf_power_curve_singleKey(self, key, mean_value, power_curve):
        values = power_curve[:,key*2:key*2+2]
        powers = []
        for m in mean_value:
            mean_values_rounded= int(m)+0.5
            row_power = np.where(values[:,0]==mean_values_rounded)[0]
            if len(row_power)!=0:
                row_power = row_power[0]
                power = values[row_power,1]
            else:
                power = 0
            powers.append(power)
        return np.array(powers)

    def get_component_value_sample(self,x, dict_, k, current_values, samples_to_delete):
        c = [dict_[j].astype("int64")[k] for j in current_values]
        c = np.hstack(c).astype("int64")
        current_vect = x[:,c]
        for j in enumerate(current_values):
            current_vect[samples_to_delete[j],j] = 0
        sum_component = np.sum(current_vect, axis = 1)
        return sum_component

    def transformPerTurbine(self, neigh_, dict_, x, power_curve,x_transf,output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]
            if key==5:
                print(key)
            if current_value.shape[1]!=0:
                current_point_level = current_value
                current_features = current_point_level[:,0:2]
                current_samples = current_point_level[:,2]
                b = np.ascontiguousarray(current_features).view(np.dtype((np.void, current_features.dtype.itemsize * current_features.shape[1])))
                _, idx = np.unique(b, return_index=True)

                unique_a = current_features[idx]
                x_tmp_u = np.zeros([n, unique_a.shape[0]])
                x_tmp_v = np.zeros([n, unique_a.shape[0]])
                for j,f in enumerate(unique_a):
                    index_f = np.where(np.all(current_features==f, axis =1))[0]
                    index_sample = current_samples[index_f]
                    absent_s = np.delete(np.arange(n), index_sample)
                    start_dim = x_transf.shape[1]
                    column_u = dict_[f[0]][f[1]]
                    x_tmp_u[:,j] = x[:,column_u]
                    x_tmp_u[absent_s,j] = 0
                    column_v = dict_[f[0]][f[1]+12]
                    x_tmp_v[:,j] = x[:,column_v]
                    x_tmp_v[absent_s,j] = 0
                sum_component_u = np.sum(x_tmp_u, axis = 1)
                sum_component_v = np.sum(x_tmp_v, axis = 1)
                wind_speed = np.sqrt(sum_component_u**2+sum_component_v**2)/len(np.unique(current_features))
                power_value = self.enel_transf_power_curve_singleKey(key, wind_speed, power_curve)
                if x_transf.shape[1]==0:
                    x_transf = power_value.reshape([n,1])
                else:
                    x_transf = np.concatenate((x_transf,power_value.reshape([n,1])), axis = 1)
                current_dim = x_transf.shape[1]
                # for i, current_v in enumerate(current_values):
                #     vect_to_append = np.array([np.arange(start_dim,current_dim)[0], k_levels[i]]).reshape([1,2])
                #     if output_dict_[current_v].shape[1]==0:
                #         output_dict_[current_v] = vect_to_append
                #     else:
                #         output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append),axis = 0)
        return x_transf, output_dict_

class EnelWindSpeedTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        n,m = x.shape
        x_transf = np.array([[]])
        dict_ = dict.fromkeys(np.arange(0,49),np.array([]))

        key = 0
        for i in range(0,m,24):
            start_dim = x_transf.shape[1]
            for j in range(i,i+12):
                if x_transf.shape[1]==0:
                    x_transf = np.sqrt(np.power(x[:,j],2)+np.power(x[:,j+12],2)).reshape([n,1])
                else:
                    x_transf = np.concatenate((x_transf,np.sqrt(np.power(x[:,j],2)+np.power(x[:,j+12],2)).reshape([n,1])), axis = 1)
            current_dim = x_transf.shape[1]
            dict_[key] = np.append(dict_[key], np.arange(start_dim,current_dim))
            key+=1

        assert (x_transf.shape[1]==m/2)

        return x_transf, dict_

class Enel_powerCurveTransformation(Transformation):
    def __init__(self):
        pass

    def transform1(self, neigh_, dict_, x, power_curve,l, x_transf,output_dict_):
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]
            if len(current_value)!=0:
                if l==0:
                    h_s = np.arange(0,len(current_value[:,0]))
                else:
                    h_s = np.arange(len(current_value[:,0])-1,len(current_value[:,0]))
                for h in h_s:
                    if l==0:
                        current_point_level = current_value[h]
                        current_values = np.array([current_point_level])[:,0]
                        k_levels = np.array([current_point_level])[:,1]
                    else:
                        current_point_level = current_value[:h+1]
                        current_values = current_point_level[:,0]
                        k_levels = current_point_level[:,1]
                    start_dim = x_transf.shape[1]

                    for k in np.unique(k_levels):
                        index_k = np.where(k_levels==k)[0]
                        values_k = current_values[index_k]

                        sum_component_u = self.get_component_value(x, dict_, k, values_k)
                        sum_component_v = self.get_component_value(x, dict_, k+12, values_k)
                        wind_speed = np.sqrt(sum_component_u**2+sum_component_v**2)/len(values_k)
                        power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                        if x_transf.shape[1]==0:
                            x_transf = power_value.reshape([n,1])
                        else:
                            x_transf = np.concatenate((x_transf,power_value.reshape([n,1])), axis = 1)
                        current_dim = x_transf.shape[1]
                    for i, current_v in enumerate(current_values):
                        vect_to_append = np.array([np.arange(start_dim,current_dim)[0], k_levels[i]]).reshape([1,2])
                        if output_dict_[current_v].shape[1]==0:
                            output_dict_[current_v] = vect_to_append
                        else:
                            output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append),axis = 0)
        return x_transf, output_dict_


    def transform(self, neigh_, dict_, x, power_curve,l, x_transf,output_dict_):
        k_levels = np.arange(0,12)
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]
            if len(current_value)!=0:
                if l==0:
                    h_s = np.arange(0,len(current_value))
                else:
                    h_s = np.arange(len(current_value)-1,len(current_value))
                for h in h_s:
                    if l==0:
                        current_values = np.array([current_value[h]])
                    else:
                        current_values = current_value[:h+1]
                    start_dim = x_transf.shape[1]
                    for k in k_levels:
                        sum_component_u = self.get_component_value(x, dict_, k, current_values)
                        sum_component_v = self.get_component_value(x, dict_, k+12, current_values)
                        wind_speed = np.sqrt(sum_component_u**2+sum_component_v**2)/len(current_values)
                        power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                        if x_transf.shape[1]==0:
                            x_transf = power_value.reshape([n,1])
                        else:
                            x_transf = np.concatenate((x_transf,power_value.reshape([n,1])), axis = 1)
                    current_dim = x_transf.shape[1]
                    for current_v in current_values:
                        vect_to_append = np.arange(start_dim,current_dim).reshape([len(np.arange(start_dim,current_dim)),1])
                        vect_to_append = np.concatenate((vect_to_append, k_levels.reshape([len(k_levels),1])), axis = 1)
                        if output_dict_[current_v].shape[1]==0:
                            output_dict_[current_v] = vect_to_append
                        else:
                            output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append),axis = 0)
        return x_transf, output_dict_

    def get_component_value(self,x, dict_, k, current_values):
        c = [dict_[j][k] for j in current_values]
        c = np.hstack(c).astype("int64")
        current_vect = x[:,c]
        sum_component = np.sum(current_vect, axis = 1)
        return sum_component

    def enel_transf_power_curve(self, key, mean_value, power_curve):
        values = power_curve[:,key*2:key*2+2]
        powers = []
        for m in mean_value:
            mean_values_rounded= int(m)+0.5
            row_power = np.where(values[:,0]==mean_values_rounded)[0]
            if len(row_power)!=0:
                row_power = row_power[0]
                power = values[row_power,1]
            else:
                power = 0
            powers.append(power)
        return np.array(powers)

    def compute_angle_matrix(self,x, num_directions = 360):
        n,m = x.shape
        x_transf = np.array([[]])
        dict_ = dict.fromkeys(np.arange(0,49),np.array([]))
        key = 0
        angle_slice = 180./num_directions
        angle_slices = np.arange(-90,90,angle_slice)
        angle_slices = angle_slices.reshape([len(angle_slices),1])
        for i in range(0,m,24):
            start_dim = x_transf.shape[1]
            for j in range(i,i+12):
                current_angle_degree = np.degrees(np.arctan2(x[:,j+12],x[:,j]))
                norm = np.abs(current_angle_degree-angle_slices)
                min_norm = np.argmin(norm, axis = 0)
                current_angle_dir = angle_slices[min_norm]

                if x_transf.shape[1]==0:
                    x_transf = current_angle_dir
                else:
                    x_transf = np.concatenate((x_transf,current_angle_dir), axis = 1)

            current_dim = x_transf.shape[1]
            dict_[key] = np.append(dict_[key], np.arange(start_dim,current_dim))
            key+=1

        assert (x_transf.shape[1]==m/2)
        return x_transf, dict_

class Enel_powerCurveTransformation_old(Transformation):
    def __init__(self):
        pass

    def transform(self, neigh_, dict_, x, power_curve,l, sum_until_k):
        k_levels = np.arange(0,12)
        x_transf = np.array([[]])
        if not sum_until_k:
            h_s = np.arange(l,l+1)
        output_dict_ = dict.fromkeys(np.arange(0,49),np.array([[]], dtype = "int64"))
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_value = values[key]#[:,0]
            if len(current_value)!=0:
                if sum_until_k:
                    h_s = np.arange(len(current_value))
                for h in h_s:
                    current_values = current_value[:h+1]
                    start_dim = x_transf.shape[1]
                    for k in k_levels:
                        sum_component_u = self.get_component_value(x, dict_, k, current_values)
                        sum_component_v = self.get_component_value(x, dict_, k+12, current_values)
                        wind_speed = np.sqrt(sum_component_u**2+sum_component_v**2)/len(current_values)
                        power_value = self.enel_transf_power_curve(key, wind_speed, power_curve)
                        if x_transf.shape[1]==0:
                            x_transf = power_value.reshape([n,1])
                        else:
                            x_transf = np.concatenate((x_transf,power_value.reshape([n,1])), axis = 1)
                    current_dim = x_transf.shape[1]
                    for current_v in current_values:
                        vect_to_append = np.arange(start_dim,current_dim).reshape([len(np.arange(start_dim,current_dim)),1])
                        vect_to_append = np.concatenate((vect_to_append, k_levels.reshape([len(k_levels),1])), axis = 1)
                        if output_dict_[current_v].shape[1]==0:
                            output_dict_[current_v] = vect_to_append
                        else:
                            output_dict_[current_v] = np.concatenate((output_dict_[current_v], vect_to_append),axis = 0)
        return x_transf, output_dict_



    def get_component_value(self,x, dict_, k, current_values):
        c = [dict_[j].astype("int64")[k] for j in current_values]
        c = np.hstack(c).astype("int64")
        current_vect = x[:,c]
        sum_component = np.sum(current_vect, axis = 1)
        return sum_component

    def enel_transf_power_curve(self, key, mean_value, power_curve):
        values = power_curve[:,key*2:key*2+2]
        powers = []
        for m in mean_value:
            mean_values_rounded= int(m)+0.5
            row_power = np.where(values[:,0]==mean_values_rounded)[0]
            if len(row_power)!=0:
                row_power = row_power[0]
                power = values[row_power,1]
            else:
                power = 0
            powers.append(power)
        return np.array(powers)



class inverseTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        x_transf = 1./x
        return x_transf

class cosTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        x_transf = np.cos(x)
        return x_transf

class sinTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        x_transf = np.sin(x)
        return x_transf

class logTrasformation(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        x_transf = np.log(x)
        return x_transf

class expTransformation(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([n, m*(self.degree)])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.exp(x**(d+1))
        return x_transf


class NullTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,X):
        #onesVec = np.ones([len(X), 1])
        #X = np.append(onesVec, X, axis=1)
        return X



class F1(Transformation):
    def __init__(self):
        self. informative = np.array([0, 1])

    def transform(self,x):
        y = (x[0]**4 - x[0]**2) * (3 + x[1])
        return y, self.informative #[-3,3]

    def expand(self,x):
        new_informatives = np.array([])
        n,m = x.shape
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())
        powers = [2,4]
        len_powers = len(powers)
        dim = len(powers)*m+len(powers)*m*m
        x_transf = np.zeros([n, dim])
        for d in range(len_powers):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,powers[d])
            new_informatives = np.append(new_informatives,self.informative+m*d)
            for key in keys_:
                dict_[key] = np.append(dict_[key], key+m*d)
        for col in range((len_powers)*m):##x^2*x, x^4*x
            x_transf[:,len_powers*m+col*m:len_powers*m+col*m+m] = x_transf[:,col].reshape([n,1])*x
            if col in new_informatives:
                new_informatives = np.append(new_informatives,np.arange(len_powers*m+col*m,len_powers*m+col*m+2))
            right_feat = col%(m)
            dict_[right_feat] = np.append(dict_[right_feat], np.arange(len_powers*m+col*m,len_powers*m+col*m+m))
            for key in keys_:
                if key!=right_feat:
                    dict_[key] = np.append(dict_[key], len_powers*m+col*m+key)

        return x_transf,new_informatives, dict_ #[-3,3]

class F2(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1])

    def transform(self,x):
        y = 2 * (x[0]**3 - x[0]) * (2*x[1] - 1) * (x[1] + 1) + (x[1]**3 - x[1] + 3)
        return y, self.informative#[-3,3]

    def expand(self,x):
        new_informatives = np.array([])
        n,m = x.shape
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())
        powers = [1,3]
        len_powers = len(powers)

        dim = m*2
        x_transf = np.zeros([n, dim])
        for d in range(len_powers): ##x e x^3
            x_transf[:,m*(d):m*(d)+m] = np.power(x,powers[d])
            new_informatives = np.append(new_informatives,self.informative+m*d)
            for key in keys_:
                dict_[key] = np.append(dict_[key], key+m*d).astype("int64")

        for count in range(m-1):
            current_dim = x_transf.shape[1]
            x_transf = np.concatenate((x_transf,x[:,count].reshape([n,1])*x[:,count+1:]), axis = 1)
            new_dim = x_transf.shape[1]
            diff_dim = new_dim-current_dim
            if count in new_informatives:
                for a in range(0,diff_dim):
                    if count+a+1 in new_informatives:
                        new_informatives = np.append(new_informatives,current_dim+a)
            dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
            for key in keys_:
                if key>count:
                    dict_[key] = np.append(dict_[key], current_dim + key-count-1)

        for count in range(m):
            for count1 in range(m):
                if count!=count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate((x_transf,x[:,count].reshape([n,1])*np.power(x[:,count1].reshape([n,1]),2)), axis = 1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                            new_informatives = np.append(new_informatives,np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        for count in range(m):
            for count1 in range(m):
                if count!=count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate((x_transf,x[:,count].reshape([n,1])*np.power(x[:,count1].reshape([n,1]),3)), axis = 1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                            new_informatives = np.append(new_informatives,np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        for count in range(m):
            for count1 in range(m):
                if count!=count1:
                    current_dim = x_transf.shape[1]
                    x_transf = np.concatenate((x_transf,np.power(x[:,count].reshape([n,1]),2).reshape([n,1])*np.power(x[:,count1].reshape([n,1]),3)), axis = 1)
                    new_dim = x_transf.shape[1]
                    if count and count1 in new_informatives:
                            new_informatives = np.append(new_informatives,np.arange(current_dim, new_dim))
                    dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
                    dict_[count1] = np.append(dict_[count1], np.arange(current_dim, new_dim))

        assert(x_transf.shape[1]==m*2+m*(m-1)/2+m*(m-1)*3)
        return x_transf,new_informatives,dict_ #[-3,3]

class F3(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1])

    def transform(self,x):
        y = -2 * (2*x[0]**2 - 1) * x[1] * math.exp(-x[0]**2 - x[1]**2)
        return y, self.informative#[-1,1]

    def expand(self,x):
        new_informatives = np.array([])
        n,m = x.shape
        x_transf = np.array([])
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())

        for count1 in range(m):
            for count2 in range(m):##x*exp(-x-x)
                if count2>count1:
                    if len(x_transf)!=0:
                        current_dim = x_transf.shape[1]
                    else:
                        current_dim = 0
                    a = x*np.exp(-np.power(x[:,count1],2)-np.power(x[:,count2],2)).reshape([n,1])
                    if count1==0 and count2==1:
                        x_transf = a
                    else:
                        x_transf = np.concatenate((x_transf,a),axis = 1)

                    new_dim = x_transf.shape[1]
                    if count1 in self.informative:
                        if count2 in self.informative:
                            new_informatives = np.append(new_informatives, np.arange(current_dim,current_dim+len(self.informative)))
                    if count1!=count2:
                        dict_[count1] = np.append(dict_[count1], np.arange(current_dim,new_dim)).astype("int64")
                    dict_[count2] = np.append(dict_[count2], np.arange(current_dim,new_dim)).astype("int64")

                    for key in keys_:
                        if key!=count2:
                            if key!=count1:
                                dict_[key] = np.append(dict_[key], current_dim+key).astype("int64")
        dim_x_transf = x_transf.shape[1]
        x_transf_copy = x_transf.copy()
        values_ = dict_.values()
        for count1 in range(m):
            current_dim = x_transf.shape[1]
            del_vect = np.arange(count1,dim_x_transf, step = m)

            x_transf_tmp = np.delete(x_transf_copy, del_vect,axis = 1)

            x_transf = np.concatenate((x_transf,np.power(x[:,count1],2).reshape([n,1])*x_transf_tmp),axis = 1)
            new_dim = x_transf.shape[1]
            dict_[count1] = np.append(dict_[count1], np.arange(current_dim,new_dim)).astype("int64")
            for key in keys_:
                if key!=count1:
                    new_values = np.array([]).astype("int64")
                    for v in values_[key]:
                        if v not in del_vect:
                            elem_less = len(del_vect[del_vect<v])
                            new_e = v-elem_less
                            new_values = np.append(new_values,  new_e)
                    new_values = np.unique(new_values)
                    dict_[key] = np.append(dict_[key], new_values+current_dim).astype("int64")
            if count1 in self.informative:
                new_informatives = np.append(new_informatives, current_dim)
        return x_transf,new_informatives, dict_

class F4(Transformation):
    def __init__(self):
       self.informative = informative = np.array([0, 1, 2])

    def transform(self,x):
        y = x[0] + (x[1]>0.5) * (x[2]>0.5)
        return y, self.informative#[0,1]

    def expand(self,x):
        new_informatives = self.informative.copy()
        n,m = x.shape
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())
        for key in keys_:
            dict_[key] = key
        x_transf = x.copy()
        for count1 in range(m):
            for count2 in range(m):
                current_dim = x_transf.shape[1]
                if count2>=count1:
                    a = (x[:,count1]>0.5).reshape([n,1])*(x[:,count2]>0.5).reshape([n,1])
                    x_transf = np.concatenate((x_transf,a),axis = 1)
                    if count1 in self.informative and count2 in self.informative:
                        new_informatives = np.append(new_informatives,current_dim)
                    if count1!=count2:
                        dict_[count1] = np.append(dict_[count1], current_dim).astype("int64")
                    dict_[count2] = np.append(dict_[count2], current_dim).astype("int64")
        return x_transf,new_informatives,dict_#[0,1]

class F5(Transformation):
    def __init__(self):
        self.informative = np.array([0, 1, 2, 3, 4])

    def transform(self,x):
        y = 10*math.sin(x[0])*x[1] + 20*(x[2] - 0.5)**2 + 10*x[3] + 5*x[4]
        return y, self.informative#[0,1]

    def expand(self,x):
        new_informatives = self.informative.copy()
        n,m = x.shape
        x_transf = x.copy()
        x_transf = np.concatenate((x_transf,np.power(x,2)),axis = 1)
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())
        for key in keys_:
            dict_[key] = np.append(dict_[key],[key,key+m]).astype("int64")
        new_informatives = np.append(new_informatives,np.arange(m,m+len(self.informative)))

        for count1 in range(m):
            current_dim = x_transf.shape[1]
            a = x*np.sin(x[:,count1]).reshape([n,1])
            x_transf = np.concatenate((x_transf,a),axis = 1)
            if count1 in self.informative:
                new_informatives = np.append(new_informatives,np.arange(current_dim, current_dim+len(self.informative)))
            dict_[count1] = np.append(dict_[count1], np.arange(current_dim,current_dim+m))
            for key in keys_:
                if key!=count1:
                    dict_[key] = np.append(dict_[key], key+current_dim).astype("int64")

        return x_transf,new_informatives,dict_#[0,1]

