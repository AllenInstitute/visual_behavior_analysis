
# -> metrics
def dprime(hit_rate,fa_rate,limits = (0.01,0.99)):
    from scipy.stats import norm
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate,limits[0],limits[1])
    fa_rate = np.clip(fa_rate,limits[0],limits[1])

    return Z(hit_rate) - Z(fa_rate)