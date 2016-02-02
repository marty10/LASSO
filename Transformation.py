from abc import ABCMeta
import abc
import math
import numpy as np
__author__ = 'Martina'


class Transformation:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def transform(self, X):
        """fit poly"""

class VariousTransformationsAllData(Transformation):
    def __init__(self, transformationList, operationsList, transformationExpPol,degreesList):
        self.trasformationList = transformationList
        self.operationsList = operationsList
        self.degreesList = degreesList
        self.transformationExpPol = transformationExpPol

    def transform(self,x):
        number_of_transf = len(self.trasformationList)
        number_of_transf_pol_exp = len(self.transformationExpPol)
        degrees = len(self.degreesList)
        n = x.shape[0]
        x_transf = np.zeros([n,number_of_transf*2+degrees*2*number_of_transf_pol_exp])
        k=0
        for operation in self.operationsList:
            indexing = 0
            for d in range(k,number_of_transf+k):
                for i in range(n):
                    x_transf[i,d] = operation(self.trasformationList[indexing](x[i,:]))
                indexing+=1
            k=number_of_transf

        k=0
        for operation in self.operationsList:
            count=0
            indexing =0
            for d in range(number_of_transf*2+k,number_of_transf*2+degrees+k):
                c = np.ones(self.degreesList[count])
                for i in range(n):
                    x_transf[i,d] = operation(self.transformationExpPol[indexing](x[i,:],c))
                count+=1
            k=degrees
        k=0
        for operation in self.operationsList:
            count=0
            indexing =1
            for d in range(number_of_transf*2+degrees*2+k,number_of_transf*2+(degrees)*2+degrees+k):
                for i in range(n):
                    x_transf[i,d] = operation(self.transformationExpPol[indexing](x[i,:]**self.degreesList[count]))
                count+=1
            k=degrees

        return x_transf


class PolinomialTransformation(Transformation):
    def __init__(self, degree, informative):
        self.degree = degree
        self.informative = informative

    def transform(self,x):
        n,m = x.shape
        new_informatives = np.array([])
        x_transf = np.zeros([n, m*(self.degree)])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
            new_informatives = np.append(new_informatives,self.informative+m*d)
        return x_transf, new_informatives

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


class expTransformationProd(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        x_transf = np.exp(x**(self.degree))*np.exp(x**self.degree-1)
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

class PolinomialKernel(Transformation):
    def __init__(self, degree):
        self.degree = degree

    def transform(self,x):
        n,m = x.shape
        K = np.zeros([m,m])
        for d in range(self.degree):
            x_transf[:,m*(d):m*(d)+m] = np.power(x,d+1)
        return x_transf


class NullTransformation(Transformation):
    def __init__(self):
        pass

    def transform(self,X):
        #onesVec = np.ones([len(X), 1])
        #X = np.append(onesVec, X, axis=1)
        return X


class F1_transf(Transformation):
    def __init__(self,informative):
         self.informative = informative

    def transform(self,x):
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

class F2_transf(Transformation):
    def __init__(self,informative):
         self.informative = informative

    def transform(self,x):
        new_informatives = np.array([])
        n,m = x.shape
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())
        powers = [1,3]
        len_powers = len(powers)
        combinations = 3
        dim = len(powers)*m+combinations*m*m
        x_transf = np.zeros([n, dim])
        for d in range(len_powers): ##x e x^3
            x_transf[:,m*(d):m*(d)+m] = np.power(x,powers[d])
            new_informatives = np.append(new_informatives,self.informative+m*d)
            for key in keys_:
                dict_[key] = np.append(dict_[key], key+m*d).astype("int64")
        k=1
        for c in range(combinations): ##x*x^2, x*x^3, x*2*x^3
            d=c
            if c==combinations-1:
                k = 2
                d=1
            for count in range(m):
                x_transf[:,len_powers*m+c*m*m+count*m:len_powers*m+c*m*m+count*m+m] = np.power(x,k)[:,count].reshape([n,1])*np.power(x,d+2)
                if count in new_informatives:
                    new_informatives = np.append(new_informatives,np.arange(len_powers*m+c*m*m+count*m,len_powers*m+c*m*m+count*m+2))
                right_feat = count%(m)
                dict_[right_feat] = np.append(dict_[right_feat], np.arange(len_powers*m+c*m*m+count*m,len_powers*m+c*m*m+count*m+m))
                for key in keys_:
                    if key!=right_feat:
                        dict_[key] = np.append(dict_[key], len_powers*m+c*m*m+count*m+key).astype("int64")

        for count in range(m): ###x*x
            current_dim = x_transf.shape[1]
            x_transf = np.concatenate((x_transf,x[:,count].reshape([n,1])*x[:,count:]), axis = 1)
            if count in new_informatives:
                new_informatives = np.append(new_informatives,np.arange(current_dim,current_dim+len(self.informative)))
            new_dim = x_transf.shape[1]
            dict_[count] = np.append(dict_[count], np.arange(current_dim, new_dim))
            for key in keys_:
                if key>count:
                    dict_[key] = np.append(dict_[key], current_dim + key)
        x_transf.astype("int64")
        return x_transf,new_informatives,dict_ #[-3,3]


class F3_transf(Transformation):
    def __init__(self,informative):
         self.informative = informative

    def transform(self,x):
        new_informatives = np.array([])
        n,m = x.shape
        x_transf = np.array([])
        dict_ = dict.fromkeys(np.arange(0,m),np.array([]))
        keys_ = np.array(dict_.keys())

        for count1 in range(m):
            for count2 in range(m):##x*exp(-x-x)
                if count2>=count1:
                    if len(x_transf)!=0:
                        current_dim = x_transf.shape[1]
                    else:
                        current_dim = 0
                    a = x*np.exp(-np.power(x[:,count1],2)-np.power(x[:,count2],2)).reshape([n,1])
                    if count1==0 and count2==0:
                        x_transf = a
                    else:
                        x_transf = np.concatenate((x_transf,a),axis = 1)

                    if count1 in self.informative:
                        if count2 in self.informative:
                            new_informatives = np.append(new_informatives, np.arange(current_dim,current_dim+len(self.informative)))
                    if count1!=count2:
                        dict_[count1] = np.append(dict_[count1], np.arange(current_dim,current_dim+m)).astype("int64")
                    dict_[count2] = np.append(dict_[count2], np.arange(current_dim,current_dim+m)).astype("int64")

                    for key in keys_:
                        if key!=count2:
                            if key!=count1:
                                dict_[key] = np.append(dict_[key], current_dim+key).astype("int64")
        dim_x_transf = x_transf.shape[1]
        x_transf_copy = x_transf.copy()
        values_ = dict_.values()
        for count1 in range(m):
            current_dim = x_transf.shape[1]
            x_transf = np.concatenate((x_transf,np.power(x[:,count1],2).reshape([n,1])*x_transf_copy),axis = 1)
            dict_[count1] = np.append(dict_[count1], np.arange(current_dim,current_dim+dim_x_transf)).astype("int64")
            for key in keys_:
                if key!=count1:
                    dict_[key] = np.append(dict_[key], values_[key]+current_dim).astype("int64")
            if count1 in self.informative:
                vect = np.arange(current_dim,current_dim+2)
                vect = np.append(vect,np.arange(current_dim+m,current_dim+m+len(self.informative)))
                new_informatives = np.append(new_informatives, vect)
        return x_transf,new_informatives, dict_

class F4_transf(Transformation):
    def __init__(self,informative):
         self.informative = informative

    def transform(self,x):

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

class F5_transf(Transformation):
    def __init__(self,informative):
         self.informative = informative

    def transform(self,x):
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


class F1(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        informative = np.array([0, 1])
        y = (x[0]**4 - x[0]**2) * (3 + x[1])
        return y, informative #[-3,3]

class F2(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        informative = np.array([0, 1])
        y = 2 * (x[0]**3 - x[0]) * (2*x[1] - 1) * (x[1] + 1) + (x[1]**3 - x[1] + 3)
        return y, informative#[-3,3]

class F3(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        informative = np.array([0, 1])
        y = -2 * (2*x[0]**2 - 1) * x[1] * math.exp(-x[0]**2 - x[1]**2)
        return y, informative#[-1,1]

class F4(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        informative = np.array([0, 1, 2])
        y = x[0] + (x[1]>0.5) * (x[2]>0.5)
        return y, informative#[0,1]

class F5(Transformation):
    def __init__(self):
        pass

    def transform(self,x):
        informative = np.array([0, 1, 2, 3, 4])
        y = 10*math.sin(x[0])*x[1] + 20*(x[2] - 0.5)**2 + 10*x[3] + 5*x[4]
        return y, informative#[0,1]