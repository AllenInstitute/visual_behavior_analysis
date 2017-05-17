from scipy.stats import norm

# -> metrics
def dprime(hit_rate,fa_rate,limits = (0.01,0.99)):
    """ calculates the d-prime for a given hit rate and false alarm rate

    https://en.wikipedia.org/wiki/Sensitivity_index

    Parameters
    ----------
    hit_rate : float
        rate of hits in the True class
    fa_rate : float
        rate of false alarms in the False class
    limits : tuple, optional
        limits on extreme values, which distort. default: (0.01,0.99)

    Returns
    -------
    d_prime

    """
    assert limits[0]>0.0, 'limits[0] must be greater than 0.0'
    assert limits[1]<1.0, 'limits[1] must be less than 1.0'
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate,limits[0],limits[1])
    fa_rate = np.clip(fa_rate,limits[0],limits[1])

    return Z(hit_rate) - Z(fa_rate)
