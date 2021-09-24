def earth_movers_distance(y_true, y_pred):
    cdf_true = y_true.cumsum(1)
    cdf_pred = y_pred.cumsum(1)
    emd = (cdf_true - cdf_pred).pow(2).mean(1).sqrt().mean()
    return emd
