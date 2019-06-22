# Weighted-Averages-Imputation

Impute using weighted averages

    from weighted_average_imputer import *
    
    df = wa_imputer(df,miss=np.nan,k=2,dec=None):
    
    #miss is the value that represents missing data in your dataset, could be np.nan, "NA", etc...
    #k: checks the k instances with the shortest cosine distance.
    #dec is the number of decimals. 0 is 0. None is full number of decimals.
