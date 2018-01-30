# Copyright (c) 2017-2018 Rizki Sasri Dwitama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as sps
import pandas as pd

__author__ = "Rizki Sasri Dwitama <sasri.darmajaya@gmail.org>"
__version__ = "1.0.2"


class DataMatrix:
    """ Dataset Loader Class """

    def __init__(self, path, delimiter=",", idx_col=0):
        """ DataMatrix constructor method """
        self.read(path, delimiter, idx_col)

    def read(self, path, delimiter=",", idx_col=0):
        """ Read data from file and convert to Pandas DataFrame Object """
        self.__data = pd.read_csv(path, delimiter=delimiter,
                           index_col=idx_col)
        self.__alab = self.__data.index
        self.__clab = self.__data.columns
        
    @property
    def random(self):
        """ Generate random matrix witn criteria's minimum and maximum value  
            and return matrix as Pandas DataFrame Object """
        c_min = self.cmin * 100
        c_max = self.cmax * 100
        alt = self.nalt
        x = np.zeros((len(c_min),alt))
        for i in range(len(c_min)):
            x[i] = np.random.randint(c_min[i], c_max[i], size=(alt))
        df = pd.DataFrame(x.transpose(), index=dm.alab, 
                          columns=dm.clab)
        return df

    @property
    def data(self):
        """ Return rating performance matrix as Pandas DataFrame Object"""
        return self.__data

    @property
    def desc(self):
        """ Describe data using statistic descriptive style"""
        return self.__data.describe()

    @property
    def nalt(self):
        """ Return alternatives number """
        return len(self.__alab)
    
    @property
    def ncrit(self):
        """ Return criterias number """
        return len(self.__clab)
    
    @property
    def alab(self):
        """ Return numpy array of alternatives label """
        return self.__alab

    @property
    def clab(self):
        """ Return numpy array of criterias label """
        return self.__clab

    @property
    def cmin(self):
        """ Return numpy array of criterias minimum value """
        ret = self.desc.loc["min"].values
        return ret

    @property
    def cmax(self):
        """ Return numpy array of criterias maximum value """
        ret = self.desc.loc["max"].values
        return ret

    @property
    def csum(self):
        """ Return numpy array of criterias total value """
        ret = np.sum(self.__data.values, axis=0)
        return ret

    @property
    def mean(self):
        """ Return numpy array of criterias mean value """
        ret = self.desc.loc["mean"].values
        return ret

    @property
    def stdv(self):
        """ Return numpy array of criterias standard deviation value """
        ret = self.desc.loc["stdv"].values
        return ret
    
    
class Normalizer(ABC):
    """ Abstract parent class for data normalization """
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _calculate(self):
        pass
    
    @property
    def rij(self):
        """ Return normalized performance ratings"""
        return pd.DataFrame(self.__rij, index=self.__alab, 
                            columns=self.__clab)
    
    def set_rij(self, xij, cstat):
        """ Normalized performance ratings setter """
        self.__alab = xij.index
        self.__clab = xij.columns
        xij = xij.values.transpose()
        rij = np.zeros((len(cstat), len(xij[0])))
        for c in range(len(cstat)):
            rij[c] = self._calculate(xij[c], cstat[c])
        self.__rij = rij.transpose()
    
    
class Vector(Normalizer):
    """ Vector Normalization Method"""

    def __init__(self, xij=None, cstat=None):
        """ Vector Normalization constructor method"""
        super(Vector, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
     
    def _calculate(self, xij, cstat):
        if (cstat == 1):
            denom = np.power(xij, 2)
            rij = xij / np.sqrt(np.sum(denom))
        else:
            # Cost
            denom = 1 / (np.power(xij, 2))
            rij = (1 / xij) / np.sqrt(np.sum(denom))
        return rij

    
class Minmax(Normalizer):
    """ Linear Transformation MinMax """

    def __init__(self, xij=None, cstat=None,
                 new_min=0, new_max=1):
        """ Linear Transformation MinMax """
        super(Minmax, self).__init__()
        self.new_min = new_min
        self.new_max = new_max
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        nomin = (xij - np.min(xij)) \
                * (self.new_max-self.new_min)
        denom = (np.max(xij)-np.min(xij)) \
                + self.new_min
        if (cstat == 1):
            rij = (nomin/denom) + self.new_min
        else:
            rij = self.new_max - (nomin/denom)
        return rij
    
    
class Lmax(Normalizer):
    """ Linear Transformation Max """

    def __init__(self, xij=None, cstat=None):
        super(Lmax, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        if (cstat == 1):
            rij = xij/np.max(xij)
        else:
            rij = 1-(xij/np.max(xij))
        return rij
    
    
class Lsum(Normalizer):
    """ Linear Transformation Sum"""

    def __init__(self, xij=None, cstat=None):
        super(Lsum, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        if (cstat == 1):
            rij = xij/np.sum(xij)
        else:
            rij = (1/xij)/np.sum((1/xij))
        return rij
    
    
class Zscore(Normalizer):
    """ Zscore Standardization """

    def __init__(self, xij=None, cstat=None):
        super(Zscore, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        zscore = sps.zscore(xij)
        if (cstat == 1):
            rij = zscore
        else:
            rij = -(zscore)
        return rij
    
    
class Gaussian(Normalizer):
    """ Gaussian Probability Distribution """

    def __init__(self, xij=None, cstat=None):
        super(Gaussian, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        zscore = sps.zscore(xij)
        if (cstat == 1):
            rij = sps.norm.cdf(zscore)
        else:
            rij = sps.norm.cdf(-zscore)
        return rij
    
    
class Sigmoid(Normalizer):
    """ Sigmoid Normalization """

    def __init__(self, xij=None, cstat=None):
        super(Sigmoid, self).__init__()
        if((xij is not None) & (cstat is not None)):
            self.set_rij(xij, cstat)
    
    def _calculate(self, xij, cstat):
        zscore = sps.zscore(xij)
        if (cstat == 1 ):
            rij = ((1-np.exp(-zscore))/(1+np.exp(-zscore)))
        else:
            rij = -((1-np.exp(-zscore))/(1+np.exp(-zscore)))
        return rij
    
    
class Weight:
    """ Performance Rate weighting parent class """

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def _calculate(self):
        pass
    
    @property
    def vij(self):
        return pd.DataFrame(self.__vij, index=self.__alab, 
                            columns=self.__clab)
    
    def set_vij(self, norm, wj):
        self.__alab = norm.rij.index
        self.__clab = norm.rij.columns
        vij = np.zeros((len(wj), len(norm.rij)))
        rij = norm.rij.values.transpose()
        for i in range(len(wj)):
            vij[i] = self._calculate(rij[i], wj[i])
        self.__vij =  vij.transpose()
        
    
class ProductWeight(Weight):
    """ Weighting method used by TOPSIS, WSM and MOORA 
        Performance Rate vij = rij * wj """

    def __init__(self, rij=None, weight=None):
        if((rij is not None) & (weight is not None)):
            self.set_vij(rij, weight)

    def _calculate(self, rij, wj):
        vij = rij * wj
        return vij
    
    
class PowerWeight(Weight):
    """ Weighting method used by WPM 
        Performance Rate vij = rij ^ wj """

    def __init__(self, rij=None, weight=None):
        if((rij is not None) & (weight is not None)):
            self.set_vij(rij, weight)
        
    def _calculate(self, rij, wj):
        vij = np.power(rij, wj)
        return vij
    
    
class MinmaxWeight(Weight):
    """ Weighting method used by VIKOR """

    def __init__(self, rij=None, weight=None):
        if((rij is not None) & (weight is not None)):
            self.set_vij(rij, weight)
        
    def _calculate(self, rij, wj):
        nomin = np.max(rij) - rij
        denom = np.max(rij) - np.min(rij)
        vij = wj * (nomin/denom)
        return vij
    
    
class Mcdm(ABC):
    """ Multi Criteria Decision Making (MCDM) 
        Abstract Parent Class 
        """
    @abstractmethod
    def __init__(self, vij, rank_mtd):
        #self.__data = vij
        self.__vij = vij.vij
        self.__rank_mtd = rank_mtd
    
    @property
    def vij(self):
        return self.__vij
    
    @property
    def rate(self):
        return self.__rate
    
    @rate.setter
    def rate(self, rate):
        self.__rate = rate
    
    @property
    def rank(self):
        return self.__rank
    
    @rank.setter
    def rank(self, rank):
        self.__rank = rank
    
    @property
    def rank_mtd(self):
        return self.__rank_mtd
    
    @rank_mtd.setter
    def rank_mtd(self, rank_mtd):
        self.__rank_mtd = rank_mtd

    @property
    def report(self):
        return self.__report
    
    def get_report(self, sortlist = "asc", alist="all", number=0):
        self.__report = self.__generate_report(sortlist, alist, number)
        return self.__report
    
    def __generate_report(self, sortlist, alist, number):
        rate = self.rate
        rank = self.rank
        vij_index = self.vij.index
        report = np.array([rate, rank]).transpose()
        rep_df = pd.DataFrame(report, index=vij_index,
                              columns=["RATE", "RANK"])
        #sorting
        if (sortlist == "desc"):
            rep_df = rep_df.sort_values(by="RANK", ascending=False)
        else:
            rep_df = rep_df.sort_values(by="RANK", ascending=True)            
        
        #selecting
        if (alist == "top"):
            rep_df = rep_df[rep_df.RANK >= len(vij_index)-number]
        elif (alist == "bottom"):
            rep_df = rep_df[rep_df.RANK <= number]
        return rep_df
    
    
class Topsis(Mcdm):
    """ TOPSIS Method """

    def __init__(self, vij=None, rank_mtd="max"):
        super(Topsis, self).__init__(vij, rank_mtd)
        self.__calculate()
        
    def __calculate(self):
        vij = self.vij.values
        pis = np.max(vij,axis=0)
        nis = np.min(vij,axis=0)
        d_max = np.sqrt(np.sum(np.power(vij-pis,2), axis=1))
        d_min = np.sqrt(np.sum(np.power(vij-nis,2), axis=1))
        rate = d_min/(d_max+d_min)
        rank = sps.rankdata(rate, method=self.rank_mtd)
        self.rate = rate
        self.rank = rank
        self.get_report(sortlist = "asc", alist="all", number=0)
        
        
class Wsm(Mcdm):
    """ Weighted Sum Model """

    def __init__(self, vij=None, rank_mtd="max"):
        super(Wsm, self).__init__(vij, rank_mtd)
        self.__calculate()
        
    def __calculate(self):
        vij = self.vij.values
        rate = np.sum(vij,axis=1)
        rank = sps.rankdata(rate, method=self.rank_mtd)
        self.rate = rate
        self.rank = rank
        self.get_report(sortlist = "asc", alist="all", number=0)
        
        
class Wpm(Mcdm):
    """ Weighted Product Model """

    def __init__(self, vij=None, rank_mtd="max"):
        super(Wpm, self).__init__(vij, rank_mtd)
        self.__calculate()
        
    def __calculate(self):
        vij = self.vij.values
        rate = np.prod(vij,axis=1)
        rank = sps.rankdata(rate, method=self.rank_mtd)
        self.rate = rate
        self.rank = rank
        self.get_report(sortlist = "asc", alist="all", number=0)
        
        
class Waspas(Mcdm):
    """ Metode WASPAS """

    def __init__(self, wsm_object, wpm_object, lamb=0.5):
        super(Waspas,self).__init__(wsm_object, "max")
        self.__wsm = wsm_object
        self.__wpm = wpm_object
        self.__lamb = lamb
        self.__calculate()
        
    def __calculate(self):
        q1 = self.__wsm.rate
        q2 = self.__wpm.rate
        rate = (self.__lamb*q1)+((1-self.__lamb)*q2)
        rank = sps.rankdata(rate, method=self.rank_mtd)
        self.rate = rate
        self.rank = rank
        self.get_report(sortlist = "asc", alist="all", number=0)
    
    
class Moora(Mcdm):
    """ MOORA Method """

    def __init__(self, vij=None, cstat=None, rank_mtd="max"):
        super(Moora, self).__init__(vij, rank_mtd)
        self.__cstat = cstat
        self.__calculate()
        
    def __calculate(self):
        vij = self.vij.values.transpose()
        cstat = self.__cstat
        y = np.zeros(len(self.vij))
        for i in range(len(cstat)):
            if (cstat[i]==1):
                y = y + vij[i]
            else:
                y = y - vij[i]        
        self.rate = y
        self.rank = sps.rankdata(self.rate, method=self.rank_mtd)
        self.get_report(sortlist = "asc", alist="all", number=0)
        
        
class Vikor(Mcdm):
    """ Vikor Method """

    def __init__(self, vij=None, new_min=0, new_max=1, rank_mtd="max"):
        super(Vikor,self).__init__(vij, rank_mtd)
        self.__new_max = new_max
        self.__new_min = new_min
        self.__calculate()
        
    def __calculate(self):
        vij = self.vij.values
        s = np.sum(vij, axis=1)
        p = np.max(vij, axis=1)
        q1 = 0.5*(((s-np.min(s)*(self.__new_max-self.__new_min)))/ \
                  ((np.max(s)-np.min(s))+self.__new_min))
        q2 = (1-0.5)*(((p-np.min(p)*(self.__new_max-self.__new_min)))/ \
                      ((np.max(p)-np.min(p))+ self.__new_min))
        q = q1 + q2
        best_rate = 1 - q
        self.rate = best_rate
        self.rank = sps.rankdata(self.rate, method=self.rank_mtd)
        self.get_report(sortlist = "asc", alist="all", number=0)
        
        
class RankSimilarity:
    """ Rank Similarity Analysis Class"""

    def __init__(self, model_list, 
                 method_label=[],
                 title="RSI TEST"):
        self.__model_list = model_list
        self.__mlab = method_label
        self.__title = title
        self.__calculate()
    
    def __df(self):
        return pd.DataFrame
    
    def __calculate(self) :
        model_list = self.__model_list
        title = self.__title
        rate_mat = []
        rank_mat = []
        for m in model_list:
            rate_mat.append(m.rate)
            rank_mat.append(m.rank)
        #calculate RSI and RSR
        rho, pval = sps.spearmanr(rank_mat, axis=1)
        rsi = np.average(rho,axis=0)
        rsr = sps.rankdata(rsi, method="max")    

        self.__rate_matrix = np.array(rate_mat)
        self.__rank_matrix = np.array(rank_mat)
        self.__corr_matrix = rho
        self.__rsi = rsi
        self.__rsr = rsr
    
    @property
    def title(self):
        """ Return analisis title """
        return self.__title
    
    @property
    def model_list(self):
        """ Return list of MCDM object"""
        return self.__model_list
    
    @property
    def rsi(self):
        """ Return Rank Similarity Index (RSI) """
        return self.__rsi
    
    @property
    def rsr(self):
        """ Return RSI rank """
        return self.__rsr
    
    @property
    def rate_matrix(self):
        """ Return method alternatives rate matrix """
        alab = self.__model_list[0].vij.index
        return pd.DataFrame(self.__rate_matrix.transpose(), 
                           index=alab, columns=self.__mlab)
    
    @property
    def rank_matrix(self):
        """ Return method alternatives rank matrix """
        alab = self.__model_list[0].vij.index
        return pd.DataFrame(self.__rank_matrix.transpose(), 
                           index=alab, columns=self.__mlab)
    
    @property
    def corr_matrix(self):
        """ Return correlation matrix """
        return pd.DataFrame(self.__corr_matrix, 
                           index=self.__mlab, 
                           columns=self.__mlab)
    
    @property
    def report(self):
        """ Return RSI report"""
        idx = self.__mlab
        col = ['RSI','RSR']
        rs = np.array([self.__rsi, self.__rsr])
        return pd.DataFrame(rs.transpose(), index=idx, 
                           columns=col).sort_values(by="RSR",
                                                    ascending=False)