#Half Normal Residual Plot Function
#Author: Chris Shartrand
#Offered free for use as is with irregular updates and bug fixes
def halfnorm(xdata,ydata,nobs):
    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import inv
    from scipy.stats import norm
    from sklearn.linear_model import LinearRegression

    xdata = np.reshape(x,(nobs,1))
    ydata = np.reshape(y,(nobs,1))
    lm = LinearRegression()
    lm.fit(xdata,ydata)
    pred = lm.predict(xdata)
    res = ydata - pred

    ones = np.ones((nobs,1))
    Xmat = np.hstack([ones,xdata])
    m = Xmat.shape[1]
    H = np.dot(np.dot(Xmat,inv(np.dot(Xmat.T,Xmat))),Xmat.T) #Hat matrix to compute standardized residuals
    RSE = np.sqrt(np.sum(res**2)/(nobs-m-1)) #root mean square error
    denom = RSE*np.sqrt(1-np.diag(H)) #make sure elementwise division always works
    std_res = np.zeros(nobs)

    for i in range(0,nobs):
        std_res[i] = res[i]/denom[i]
    abs_res = np.absolute(std_res) #generating the halfnormal values
    hold_outlier = []
    hold_i = []
    for i in range(0,nobs):
        if abs_res[i] >= 2.0: #here I have chosen 2.5 SD's away from mean to indicate a possible outlier. Change as you wish
            hold_outlier.append(abs_res[i])
            hold_i.append(i)

    outlier_info = np.column_stack((hold_outlier,hold_i))
    outlier_info = sorted(outlier_info, key=lambda x: x[0])
    abs_res.sort()


    invnorm = np.zeros(nobs)
    for i in range(1,nobs):
        invnorm[i] = norm.ppf((i+nobs+.5)/(2*nobs+0.0+9./8)) #chosen inverse standard normal formulas may vary. This is generally stable

    yloc = []
    for i in range(0,len(outlier_info)):
        yloc.append(norm.ppf((nobs-i-1+nobs+.5)/(2*nobs+0.0+9./8)))
    yloc.sort() #necessary to generate values for annotating the datapoints

    fig, ax = plt.subplots()
    ax.scatter(invnorm,abs_res)
    for i in enumerate(outlier_info):
        j = i[0]
        info = outlier_info[j]
        ax.annotate(int(info[1]),(yloc[j],info[0]))
    ax.set_xlabel('Half Normal Quantities')
    ax.set_ylabel('Sorted Values');
