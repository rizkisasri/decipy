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

""" 
A collection of Multi Criteria Decision Making (MCDM) functions 
for python.  The function names appear below.

Disclaimers :  The function list is obviously incomplete and, worse, the
functions are not optimized.  All functions have been tested (some more
so than others), but they are far from bulletproof.  Thus, as with any
free software, no warranty or guarantee is expressed or implied.

"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as sps
import pandas as pd

__author__ = "Rizki Sasri Dwitama <sasri.darmajaya@gmail.org>"
__version__ = "1.0.1"
__all__ = ['norm_vector', 'norm_minmax', 'norm_minmax2', 'norm_max',
           'norm_sum','norm_zscore', 'norm_gaussian', 'norm_sigmoid',
           'norm_softmax','weight_multi', 'weight_power', 'weight_minmax', 
           'wsm', 'wpm', 'waspas', 'moora', 'topsis', 'vikor', 
           'random_matrix', 'data_matrix', 'rs_analysis', 'rs_simulation' ]


def norm_vector(xij, cstat):
    """ 
    Vector normalization function 
    -----------------------------
        In this method, each performance rating of the
        decision matrix is divided by its norm. This method 
        has the advantage of converting all attributes into 
        dimensionless measurement unit, thus making inter-attribute 
        comparison easier. But it has the drawback of having non-equal 
        scale length leading to difficulties in straightforward 
        comparison [3][4].
    
    Parameter 
    ---------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
    
    See also
    --------
    norm_max, norm_minmax, norm_minmax2, norm_sum
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_vector(xij, cstat)
              C1        C2        C3
    A1  0.093982  0.605303  0.556480
    A2  0.354605  0.439318  0.757900
    A3  0.930281  0.663783  0.340466
    
    References
    ---------
        [1] ÇELEN, Aydın. 2014. "Comparative Analysis of 
            Normalization Procedures in TOPSIS Method: 
            With an Application to Turkish Deposit Banking Market." 
            INFORMATICA 25 (2): 185–208
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)
        [3] Yoon, K.P. and Hwang, C.L., “Multiple Attribute
            Decision Making: An Introduction”, SAGE publications, 
            London, 1995.
        [4] Hwang, C.L. and K. Yoon, “Multiple Attribute Decision 
            Making: Methods and Applications”, Springer-Verlag, 
            New York, 1981.
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1):
            # Benefit Criteria
            denom = np.power(xij[c], 2)
            rij[c] = xij[c] / np.sqrt(np.sum(denom))
        else:
            # Cost Criteria
            denom = 1 / (np.power(xij[c], 2))
            rij[c] = (1 / xij[c]) / np.sqrt(np.sum(denom))
    result = pd.DataFrame(rij.transpose(),
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_minmax(xij, cstat):
    """ 
    Linear Minmax normalization function
    ----------------------------------
        This method considers both the maximum and minimum 
        performance ratings of criterias during calculation.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
    
    See also
    --------
    norm_vector, norm_max, norm_minmax2, norm_sum
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_minmax(xij, cstat)
              C1        C2        C3
    A1  0.000000  0.739469  0.704785
    A2  0.311639  0.000000  1.000000
    A3  1.000000  1.000000  0.000000
    
    References
    ---------
        [1] ÇELEN, Aydın. 2014. "Comparative Analysis of 
            Normalization Procedures in TOPSIS Method: 
            With an Application to Turkish Deposit Banking Market." 
            INFORMATICA 25 (2): 185–208
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)    
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1):
            # Benefit Criteria
            nomin = (xij[c] - np.min(xij[c]))
            denom = (np.max(xij[c])-np.min(xij[c]))
            rij[c] = (nomin/denom)
        else:
            # Cost Criteria
            nomin = (np.max(xij[c]) - xij[c])
            denom = (np.max(xij[c])-np.min(xij[c]))
            rij[c] = (nomin/denom)
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_minmax2(xij, cstat, new_min=0, new_max=1):
    """ 
    Linear Minmax normalization function 
    (with new_min and new_max values)
    ------------------------------------
        This method considers both the maximum and minimum 
        performance ratings of criterias during calculation. 
        In this method, we can sets both new minimum and new maximum 
        value.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
        new_min : integer or floating point number (optional)
            New criteria normalized minimum number, default is 0.
        new_max : integer or floating point number (optional)
            New criteria normalized maximum number, default is 1.
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range new_min 
            to new_max number or 0.0 to 1.0 by default.
    
    See also
    --------
    norm_vector, norm_max, norm_minmax, norm_sum
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> norm_minmax2(xij, cstat, 10, 100)
            C1        C2        C3
    A1   10.00   10.0000  100.0000
    A2  100.00  100.0000   55.5935
    A3   44.67   33.4838   10.0000

    References
    ---------
        [1] Han, Jiawei, Micheline Kamber, and Jian Pei. 2012. 
            Data Mining Concepts and Techniques Third Edition. 
            Waltham: Elsevier Inc 
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        nomin = (xij[c] - np.min(xij[c])) * (new_max-new_min)
        denom = (np.max(xij[c])-np.min(xij[c])) 
        if (cstat[c] == 1):
            # Benefit Criteria
            rij[c] = (nomin/denom) + new_min
        else:
            # Cost Criteria
            rij[c] = new_max - (nomin/denom)
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_max(xij, cstat):
    """ 
    Linear Max normalization function.
    ----------------------------------
        This method divides the performance ratings of each criteria 
        by the maximum performance rating for that criteria.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
    
    See also
    --------
    norm_vector, norm_minmax, norm_minmax2, norm_sum
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_max(xij, cstat)
              C1        C2        C3
    A1  0.101025  0.911899  0.388179
    A2  0.381180  0.661841  0.550777
    A3  1.000000  1.000000  0.000000
    
    References
    ---------
        [1] ÇELEN, Aydın. 2014. "Comparative Analysis of 
            Normalization Procedures in TOPSIS Method: 
            With an Application to Turkish Deposit Banking Market." 
            INFORMATICA 25 (2): 185–208
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1):
            # Benefit Criteria
            rij[c] = xij[c]/np.max(xij[c])
        else:
            # Cost Criteria
            rij[c] = 1-(xij[c]/np.max(xij[c]))
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_sum(xij, cstat):
    """ 
    Linear Sum normalization function.
    ----------------------------------
        This method divides the performance ratings of each
        attribute by the sum of performance ratings for that
        attribute.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
    
    See also
    --------
    norm_vector, norm_minmax, norm_minmax2, norm_max
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_sum(xij, cstat)
              C1        C2        C3
    A1  0.068159  0.354309  0.336273
    A2  0.257171  0.257151  0.457988
    A3  0.674670  0.388540  0.205739
    
    References
    ---------
        [1] ÇELEN, Aydın. 2014. "Comparative Analysis of 
            Normalization Procedures in TOPSIS Method: 
            With an Application to Turkish Deposit Banking Market." 
            INFORMATICA 25 (2): 185–208
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1):
            # Benefit Criteria
            rij[c] = xij[c]/np.sum(xij[c])
        else:
            # Cost Criteria
            rij[c] = (1/xij[c])/np.sum((1/xij[c]))
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_zscore(xij, cstat):
    """ 
    Z-Score normalization function.
    -------------------------------
        In z-score normalization (or zero-mean normalization), 
        the values for an attribute, A, are normalized based on 
        the mean (i.e., average) and standard deviation of A. This
        method of normalization is useful when the actual minimum 
        and maximum of attribute A are unknown, or when there are 
        outliers that dominate the min-max normalization.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range -3.0 to 3.0.
    
    See also
    --------
    norm_gaussian, norm_sigmoid, norm_softmax
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_zscore(xij, cstat)
              C1        C2        C3
    A1 -1.046482  0.376908  0.325437
    A2 -0.300566 -1.368901  1.029157
    A3  1.347047  0.991993 -1.354594
    
    References
    ---------
        [1] Han, Jiawei, Micheline Kamber, and Jian Pei. 2012. 
            Data Mining Concepts and Techniques Third Edition. 
            Waltham: Elsevier Inc
        [2] Spatz, Chris. 2008. Basic Statistics: Tales of 
            Distributions, Ninth Edition. Belmont: Thomson 
            Learning
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        zscore = sps.zscore(xij[c])
        if (cstat[c] == 1):
            # Benefit Criteria
            rij[c] = zscore
        else:
            # Cost Criteria
            rij[c] = -(zscore)
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_gaussian(xij, cstat):
    """ 
    Gaussian normalization function
    -------------------------------
        This normalization method applies gaussian probability
        function.

    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0.0 to 1.0.
    
    See also
    --------
    norm_zscore, norm_sigmoid, norm_softmax
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_gaussian(xij, cstat)
            C1      C2      C3
    A1  0.3266  0.9092  0.7540
    A2  0.1745  0.3949  0.7663
    A3  0.9171  0.1424  0.0787
    
    References
    ---------
        [1] Barrow, Michael. 2017. Statistics for Economics, 
            Accounting and Business Studies Seventh Edition. 
            Pearson Pearson Education Limited. United Kingdom
        [2] Spatz, Chris. 2008. Basic Statistics: Tales of 
            Distributions, Ninth Edition. Belmont: Thomson 
            Learning
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        zscore = sps.zscore(xij[c])
        if (cstat[c] == 1):
            # Benefit Criteria
            rij[c] = sps.norm.cdf(zscore)
        else:
            # Benefit Criteria
            rij[c] = sps.norm.cdf(-zscore)
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def norm_softmax(xij, cstat):
    """ 
    Softmax normalization function
    ------------------------------
        The hyperbolic tangent function, tanh, limits the range 
        of the normalized data to values between −1 and 1. 
        The hyperbolic tangent function is almost linear near the 
        mean, but has a slope of half that of the sigmoid function. 
        Like sigmoid, it has smooth, monotonic nonlinearity at both 
        extremes.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) 
            or (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range -1 to 1.
    
    See also
    --------
    norm_zscore, norm_gaussian, norm_sigmoid
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_softmax(xij, cstat)
              C1        C2        C3
    A1 -1.046482  0.376908  0.325437
    A2 -0.300566 -1.368901  1.029157
    A3  1.347047  0.991993 -1.354594
    
    References
    ---------
        [1] "Sigmoid Function" 
            https://en.wikipedia.org/wiki/Sigmoid_function
    """
    
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1 ):
            # Benefit Criteria
            zscore = sps.zscore(xij[c])
            rij[c] = ((1-np.exp(-zscore))/(1+np.exp(-zscore)))
        else:
            # Cost Criteria
            zscore = -(sps.zscore(xij[c]))
            rij[c] = ((1-np.exp(-zscore))/(1+np.exp(-zscore)))
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)

def norm_sigmoid(xij, cstat):
    """ 
    Sigmoid normalization function 
    ------------------------------
        The sigmoid function limits the range of the normalized data 
        to values between 0 and 1. The sigmoid function is almost 
        linear near the mean and has smooth nonlinearity at both 
        extremes, ensuring that all data points are within a limited 
        range. This maintains the resolution of most values within 
        a standard deviation of the mean.
    
    Parameter 
    ----------
        xij : Pandas DataFrame object
            Alternative performance rating matrix. 
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
    
    Return
    ------
        result : Pandas DataFrame object
            Return normalized performance rating, range 0 to 1.
    
    See also
    --------
    norm_zscore, norm_gaussian, norm_softmax
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> dcp.norm_sigmoid(xij, cstat)
            C1      C2      C3
    A1  0.4853  0.4568  0.3277
    A2  0.7778  0.2444  0.3328
    A3  0.2325  0.7862  0.8044

    References
    ---------
        [1] "Sigmoid Function" 
            https://en.wikipedia.org/wiki/Sigmoid_function
    """
    alab = xij.index
    clab = xij.columns
    xij = xij.values.transpose()
    rij = np.zeros((len(cstat), len(xij[0])))
    for c in range(len(cstat)):
        if (cstat[c] == 1 ):
            # Benefit Criteria
            zscore = sps.zscore(xij[c])
            rij[c] = (1/(1+np.exp(-zscore)))
        else:
            # Cost Criteria
            zscore = -(sps.zscore(xij[c]))
            rij[c] = (1/(1+np.exp(-zscore)))
    result = pd.DataFrame(rij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def weight_multi(rij, wj):
    """ Product weighting function"""
    alab = rij.index
    clab = rij.columns
    vij = np.zeros((len(wj), len(rij)))
    rij = rij.values.transpose()
    for i in range(len(wj)):
        vij[i] = rij[i] * wj[i]
    result = pd.DataFrame(vij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def weight_power(rij, wj):
    """ Power weighting function"""
    alab = rij.index
    clab = rij.columns
    vij = np.zeros((len(wj), len(rij)))
    rij = rij.values.transpose()
    for i in range(len(wj)):
        vij[i] = np.power(rij[i],wj[i])
    result = pd.DataFrame(vij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def weight_minmax(rij, wj):
    """ Minmax weighting function"""
    alab = rij.index
    clab = rij.columns
    vij = np.zeros((len(wj), len(rij)))
    rij = rij.values.transpose()
    for i in range(len(wj)):
        nomin = np.max(rij[i]) - rij[i]
        denom = np.max(rij[i]) - np.min(rij[i])
        vij[i] = wj[i] * (nomin/denom)
    result = pd.DataFrame(vij.transpose(), 
                          index=alab, 
                          columns=clab)
    return np.round(result,4)


def wsm(data, cweight, rank_method="max"):
    """ 
    Weighted Sum Model / Simple Additive Weighting 
    ----------------------------------------------
        The assumption that governs this model is 
        the additive utility assumption.  The basic logic 
        of the WSM/SAW method is to obtain a weighted sum of 
        the performance ratings of each alternative over 
        all attributes.
            
    Parameter 
    ----------
        data : Pandas DataFrame object
            Normalized (Linear Max) performance rating matrix. 
        cweight : Array like
            Weight of each criterias.
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking .
    
    See also
    --------
        wpm, waspas, moora
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> rij = dcp.norm_max(xij, cstat)
    >>> wsm = dcp.wsm(rij, weight)
    >>> wsm
          RATE  RANK
    A1  0.4336   2.0
    A2  0.7313   3.0
    A3  0.4140   1.0

    references
    ----------
        [1] Triantaphyllou, E., Mann, S.H. 1989. 
            "An Examination of The Effectiveness of Multi-dimensional 
            Decision-making Methods: A Decision Making Paradox." 
            Decision Support Systems (5(3)): 303–312.
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)
        [3] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    vij = weight_multi(data, cweight)
    alab = data.index
    rate = np.sum(vij.values, axis=1)
    rank = sps.rankdata(rate, method=rank_method)
    rest = np.array([rate, rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['RATE','RANK'])
    return np.round(result,4)


def wpm(data, cweight, rank_method="max"):
    """ 
    Weighted Product Model 
    ----------------------
        The weighted product model (WPM) is very
        similar to the WSM. The main difference is that
        instead of addition in the model there is 
        multiplication. Each alternative is compared 
        with the others by multiplying a number of ratios, 
        one for each criterion. Each ratio is raised to the 
        power equivalent of the relative weight of the 
        corresponding criterion.
        
    Parameter 
    ----------
        data : Pandas DataFrame object
            Performance rating matrix. (Normalized is good) 
        cweight : Array like
            Weight of each criterias.
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking.
            
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> wpm = dcp.wpm(xij, weight)
    >>> wpm
          RATE  RANK
    A1  0.6168   3.0
    A2  0.4114   2.0
    A3  0.1868   1.0
    
    See also
    --------
        wsm, waspas, moora
    
    references
    ----------
        [1] Triantaphyllou, E., Mann, S.H. 1989. 
            "An Examination of The Effectiveness of Multi-dimensional 
            Decision-making Methods: A Decision Making Paradox." 
            Decision Support Systems (5(3)): 303–312.
        [2] Chakraborty, S., and C.H. Yeh. 2012. 
            "Rank Similarity based MADM Method Selection." 
            International Conference on Statistics in Science, 
            Business and Engineering (ICSSBE2012)
        [3] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    vij = weight_power(data, cweight)
    alab = data.index
    rate = np.prod(vij.values, axis=1)
    rank = sps.rankdata(rate, method=rank_method)
    rest = np.array([rate, rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['RATE','RANK'])
    return np.round(result,4)


def waspas(wsm, wpm, lamb=0.5, rank_method="max"):
    """ 
    Weighted Aggregated Sum Product Asessment (WASPAS) 
    --------------------------------------------------
        Supposing the increase of ranking accuracy and, 
        respectively, the effectiveness of decision making, 
        a joint criterion of WSM and WPM for determining 
        a total importance of alternatives, called the weighted 
        aggregated sum product assessment (WASPAS) method.
    
    Parameter 
    ----------
        wsm : Pandas DataFrame object
            WSM generated alternatives overal rating and ranking.
        wpm : Pandas DataFrame object
            WPM generated alternatives overal rating and ranking.
        lamb : float
            Initial criteria accuracy (0.0 to 1.0) when lamb=0
            WASPAS is transformed to WPM; and when lamb=1,
            WASPAS is transformed to WSM.
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking .
    
    See also
    --------
        wpm, waspas, moora

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> rij = dcp.norm_max(xij, cstat)
    >>> wsm = dcp.wsm(rij, weight)
    >>> wpm = dcp.wpm(xij, weight)
    >>> waspas = dcp.waspas(wsm, wpm, lamb=0.5)
    >>> waspas
            Q1      Q2    RATE  RANK
    A1  0.5048  0.7034  0.6041   2.0
    A2  0.4900  0.2452  0.3676   1.0
    A3  0.5728  0.6718  0.6223   3.0

    references
    ----------
        [1] Zavadskas, E. K., Turskis, Z., Antucheviciene, J., 
            & Zakarevicius, A. 2012. "Optimization of Weighted 
            Aggregated Sum Product Assessment." Elektronika ir 
            elektrotechnika (122(6), 3-6.)
        [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    q1 = wsm['RATE']
    q2 = wpm['RATE']
    alab = wsm.index
    rate = (lamb*q1) + ((1-lamb)*q2)
    rank = sps.rankdata(rate, method=rank_method)
    rest = np.array([q1, q2, rate,rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['Q1', 'Q2', 
                                   'RATE','RANK'])
    return np.round(result,4)


def moora(data, cweight, cstat, rank_method="max"):
    """ 
    Multi-Objective Optimization on the Basis of
    Ratio Analysis (MOORA) Ratio System 
    --------------------------------------------
        The MOORA method consists of 2 parts: the ratio system 
        and the reference point approach. This function is based on
        MOORA Ratio System.
        
    Parameter 
    ----------
        data : Pandas DataFrame object
            Vector Normalized performance rating matrix 
            Note: reset all cstat to benefit when normalizing 
                  performance rating matrix. 
        cweight : Array like
            Weight of each criterias.
        cstat : Array like
            List of criteria benefitial status (benefit = 1) or 
            (cost = 0)
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking .
    
    See also
    --------
        wsm, wpm, waspas
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> cstat_reset = np.ones(len(clab))
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> rij = norm_vector(xij, cstat_reset)
    >>> moora = moora(rij, weight, cstat)
    >>> moora
           BEN     COS    RATE  RANK
    A1  0.5218  0.2091  0.3127   3.0
    A2  0.3219  0.0933  0.2286   2.0
    A3  0.2532  0.2423  0.0109   1.0
    
    references
    ----------
        [1] Brauers, Willem K., and Edmundas K. Zavadskas. 2009. 
            "Robustness of the multi‐objective MOORA method with 
            a test for the facilities sector." Ukio Technologinis 
            ir Ekonominis (15:2): 352-375.
        [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    vij = (weight_multi(data, cweight)).values
    vij = vij.transpose()
    alab = data.index
    ben = np.zeros(len(data.values))
    cos = np.zeros(len(data.values))
    y = np.zeros(len(data.values))
    for i in range(len(cstat)):
        if (cstat[i]==1):
            ben = ben + vij[i]
            y = y + vij[i]
        else:
            cos = cos + vij[i]
            y = y - vij[i]
    rate = y
    rank = sps.rankdata(rate, method=rank_method)
    rest = np.array([ben, cos, rate, rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['BEN','COS',
                                   'RATE','RANK'])
    return np.round(result,4)


def topsis(data, cweight, rank_method="max"):
    """ 
    Technique for Order Preferences by Similarity 
    to an Ideal Solution (TOPSIS) 
    ---------------------------------------------
        TOPSIS applies a simple concept of maximizing distance
        from the negative-ideal solution and minimizing the 
        distance from the positive ideal solution.  The chosen 
        alternative must be as close as possible to the ideal 
        solution and as far as possible from the negative-ideal 
        solution.
    
    Parameter 
    ----------
        data : Pandas DataFrame object
            Normalized (Vector) performance rating matrix. 
        cweight : Array like
            Weight of each criterias.
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking .
    
    See also
    --------
        wsm, wpm, waspas, moora, vikor
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> rij = dcp.norm_vector(xij, cstat)
    >>> topsis = dcp.topsis(rij, weight)
    >>> topsis
            D+      D-    RATE  RANK
    A1  0.0042  0.2745  0.9849   3.0
    A2  0.2743  0.0152  0.0525   1.0
    A3  0.1598  0.1163  0.4211   2.0

    references
    ----------
        [1] Hwang, C.L., and K. Yoon. 1981. "Multiple attribute 
            decision making, methods and applications." Lecture 
            Notes in Economics and Mathematical Systems 
            (Springer-Verlag) 186
        [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    vij = weight_multi(data, cweight)
    alab = data.index
    pis = np.max(vij,axis=0)
    nis = np.min(vij,axis=0)
    dmax = np.sqrt(np.sum(np.power(vij-pis,2), axis=1))
    dmin = np.sqrt(np.sum(np.power(vij-nis,2), axis=1))
    rate = dmin/(dmax+dmin)
    rank = sps.rankdata(rate, method=rank_method)
    rest = np.array([dmax, dmin, rate,rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['D+','D-',
                                   'RATE','RANK'])
    return np.round(result,4)


def vikor(data, cweight, rank_method="max"):
    """ 
    VlseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR) 
    ------------------------------------------------------------
        This method focuses on ranking and selecting from a set of 
        alternatives in the presence of conflicting criteria. 
        It introduces the multicriteria ranking index based on 
        the particular measure of “closeness” to the 
        “ideal” solution (Opricovic 1998).

    Parameter 
    ----------
        data : Pandas DataFrame object
            Normalized (sum) performance rating matrix. 
        cweight : Array like
            new of each criterias.
        rank_method : string
            Ranking method ('max', 'min', 'average').
    
    Return
    ------
        result : Pandas DataFrame object
            Return alternatives overal rating and ranking .
    
    See also
    --------
        wsm, wpm, waspas, moora, topsis
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import decipy.pro as dcp
    >>> x = np.random.random((3,3))
    >>> alab = ['A1','A2','A3'] # alternative label
    >>> clab = ['C1','C2','C3'] # criteria label
    >>> cstat = [1,1,0]
    >>> weight = [0.3333,0.3333,0.3333]
    >>> xij = pd.DataFrame(x, index=alab, columns=clab)
    >>> rij = dcp.norm_vector(xij, cstat)
    >>> vikor = dcp.vikor(rij, weight)
    >>> vikor
             S       P    RATE  RANK
    A1  0.3364  0.3333  0.5758   2.0
    A2  0.6918  0.3333  1.0000   3.0
    A3  0.2729  0.2729  0.0000   1.0

    references
    ----------
        [1] Hwang, C.L., and K. Yoon. 1981. "Multiple attribute 
            decision making, methods and applications." Lecture 
            Notes in Economics and Mathematical Systems 
            (Springer-Verlag) 186
        [2] “Ranking”, http://en.wikipedia.org/wiki/Ranking
    """
    vij = weight_minmax(data, cweight)
    new_min=0
    new_max=1
    alab = data.index
    s = np.sum(vij, axis=1)
    p = np.max(vij, axis=1)
    q1 = 0.5*(((s-np.min(s)*(new_max-new_min)))/ \
              ((np.max(s)-np.min(s))+new_min))
    q2 = (1-0.5)*(((p-np.min(p)*(new_max-new_min)))/ \
                  ((np.max(p)-np.min(p))+ new_min))
    q = q1 + q2
    best = 1 - q
    rank = sps.rankdata(best, method=rank_method)
    rest = np.array([s, p, q, rank])
    result = pd.DataFrame(rest.transpose(), 
                          index=alab, 
                          columns=['S','P',
                                   'RATE','RANK'])
    return np.round(result,4)


def data_matrix(path, delimiter=",", idx_col=0):
        """ 
        Read data from file and convert to Data Frame Object 
        This function simplify pandas read_csv function.
        """
        result = pd.read_csv(path, delimiter=delimiter,
                           index_col=idx_col)
        return result

    
def random_matrix(cmin, cmax, clab, alab, nalt):
    cmin = cmin * 100
    cmax = cmax * 100
    x = np.zeros((len(cmin), nalt))
    for i in range(len(cmin)):
        x[i] = np.random.randint(cmin[i], cmax[i], size=(nalt)) / 100
    df = pd.DataFrame(x.transpose(), index=alab, 
                      columns=clab)
    return np.round(df,4)


def rs_analysis(model_list, model_label, title) :
    rate_mat = []
    rank_mat = []
    result = {}
    alt_index = model_list[0].index
    for m in model_list:
        rate_mat.append(m['RATE'])
        rank_mat.append(m['RANK'])
    #RSI and RSR
    rho, pval = sps.spearmanr(rank_mat, axis=1)
    rsi = np.average(rho,axis=0)
    
    rate_df = pd.DataFrame(np.array(rate_mat).transpose(),
                           index=alt_index, columns=model_label)
    rank_df = pd.DataFrame(np.array(rank_mat).transpose(),
                           index=alt_index, columns=model_label)
    corr_df = pd.DataFrame(rho, index=model_label, columns=model_label)
    corr_df['RSI'] = np.round(rsi,4)
    corr_df['RSR'] = sps.rankdata(rsi, method="max")
    
    result['rate_matrix'] = np.round(rate_df, 4)
    result['rank_matrix'] = np.round(rank_df, 4)
    result['corr_matrix'] = np.round(corr_df, 4)
    
    return result

def rs_simulation(model_list, model_label, trials, title):
    
    tr_mat = np.zeros((len(trials),len(model_label)))
    trp_mat = np.zeros((len(trials),len(model_label)))
    gsi_mat = np.zeros((len(trials),len(model_label)))
    tsi_mat = np.zeros((len(trials),len(model_label)))
    
    print("Simulation started !, get some coffee !")
    ei_mat = np.zeros((np.max(trials),len(model_label)))
    er_mat = np.zeros((np.max(trials),len(model_label)))

    for t in range(len(trials)):
        if (t == 0):
            prev_event = 0
            next_event = trials[t]
        else:
            prev_event = trials[t-1]+1
            next_event = trials[t] - prev_event
        for e in range(next_event):
            rsi_report = rs_analysis(model_list, model_label, title)
            ei_mat[e+prev_event] = rsi_report['corr_matrix']['RSI']
            er_mat[e+prev_event] = rsi_report['corr_matrix']['RSR']
            #print("Trial #%i, %i Event" % (t+1,e+prev_event))
            
        tr_all = np.count_nonzero(er_mat == len(model_label),axis=0)
        tr_mat[t] = tr_all
        trp_mat[t] = tr_all/trials[t]
        gsi_mat[t] = np.average(ei_mat,axis=0)
        tsi_mat[t] = np.sum(ei_mat,axis=0)
        
        print("Trial #%i, %i event, data collected !" % (t+1,trials[t]))
    
    print("Simulation Completed !")
    result = {"TR": pd.DataFrame(np.array(tr_mat, dtype="int").transpose(),
                           index=model_label, columns=trials), 
              "TRP": pd.DataFrame(np.array(trp_mat).transpose(),
                           index=model_label, columns=trials), 
              "GSI": pd.DataFrame(np.array(gsi_mat).transpose(),
                           index=model_label, columns=trials), 
              "TSI": pd.DataFrame(np.array(tsi_mat).transpose(),
                           index=model_label, columns=trials)}
    return result