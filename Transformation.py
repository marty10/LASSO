from abc import ABCMeta
import abc
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import numpy.linalg as li


class Transformation:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """fit poly"""

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

    def nearest_products(self, neigh_, dict_, x):
        x_transf = np.array([[]])
        #output_dict_ = dict.fromkeys(np.arange(0,49),np.array([]))
        n = x.shape[0]
        keys_ = (list)(neigh_.keys())
        values = (list)(neigh_.values())
        for key in keys_:
            current_values = values[key]
            v = [dict_[j] for j in current_values]
            k_levels = dict_[key]
            v = np.hstack(v).astype("int64")
            for k in k_levels:
                prod = x[:,k].reshape([n,1])*x[:,v]
                if x_transf.shape[1]==0:
                    x_transf = prod
                else:
                    x_transf = np.concatenate((x_transf,prod), axis = 1)
        return x_transf


class EnelTransformation(Transformation):
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
                    x_transf = np.power(np.sqrt(np.power(x[:,j],2)+np.power(x[:,j+12],2)).reshape([n,1]),3)
                else:
                    x_transf = np.concatenate((x_transf,np.power(np.sqrt(np.power(x[:,j],2)+np.power(x[:,j+12],2)).reshape([n,1]),3)), axis = 1)

            x_transf = x_transf*59.3/100
            current_dim = x_transf.shape[1]
            dict_[key] = np.append(dict_[key], np.arange(start_dim,current_dim))
            key+=1

        assert (x_transf.shape[1]==m/2)

        return x_transf, dict_


class VariousTransformationsAllData(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n = x.shape[0]
        x_transf = np.array([])
        for d in range(2,self.degree):
            poly = PolynomialFeatures(degree=d)
            if d==2:
                x_transf=np.sum(poly.fit_transform(x)[:,1:],axis=1).reshape([n,1])
            else:
                x_transf = np.concatenate((x_transf,np.sum(poly.fit_transform(x)[:,1:],axis=1).reshape([n,1])), axis = 1)
        x_transf = np.concatenate((x_transf,np.sum(np.sin(x),axis=1).reshape([n,1])), axis = 1)
        x_transf = np.concatenate((x_transf,np.sum(np.cos(x),axis=1).reshape([n,1])), axis = 1)
        #x_transf = np.concatenate((x_transf,np.sum(np.log(x),axis=1).reshape([n,1])), axis = 1)
        x_transf = np.concatenate((x_transf,np.sum(np.exp(x),axis=1).reshape([n,1])), axis = 1)
        x_transf = np.concatenate((x_transf,np.sum(1./x,axis=1).reshape([n,1])), axis = 1)
        return x_transf

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

class PolinomialTransformation(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([n, m*(self.degree)])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
        return x_transf

class PolinomialTransformationExp(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([n, m*(self.degree)*2])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
        count = 0
        for d in range(self.degree, self.degree*2):
            x_transf[:,m*(d):m*(d)+m] = np.exp(x**count)
            count+=1
        return x_transf

class PolinomialTransformationProd(Transformation):
    def __init__(self, degree, informative):
        self.degree = degree
        self.informative = informative

    def transform(self,x):
        n,m = x.shapew
        for d in range(self.degree):
            x_transf = np.power(x**(d))*np.exp(x**self.degree-1)
        return x_transf

class expTransformationProd(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.exp(x**(self.degree))*np.exp(x**self.degree-1)
        return x_transf


class AllTransformation(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.zeros([n, m*(self.degree)*2+3*m])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
        count = 1
        for d in range(self.degree, self.degree+m):
            x_transf[:,m*(d):m*(d)+m] = np.exp(x**count)
            count+=1
        x_transf[:, m*(self.degree)*2+m: m*(self.degree)*2+2*m] = np.sin(x)
        x_transf[:, m*(self.degree)*2+2*m: m*(self.degree)*2+3*m] = np.exp2(x)
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

