import numpy as np
import pandas as pd


def cosDist(yes,no):  # inputs are both vectors , 1 row compared to 1 row
    # function to calculate cosine distance
    num = np.sum(yes*no)
	
    den1 = np.sqrt(np.sum(np.power(yes,2)))
    den2 = np.sqrt(np.sum(np.power(no,2)))
    den = den1*den2
	
    ret = 1 - (num/den)
	
    return(ret)

def weightedAverage(yes,no_na,miss,k,dec):
    # remove nan value from row
    r = (yes.dropna())
    
    dists = []  # stores cosine distance
    for no in no_na.iterrows():
        dists.append(cosDist(r,no[1]))
    # index of top k distances
    top_indexes = np.argsort(dists)[-k:]
    # list of top k distances
    D = np.array([dists[i] for i in top_indexes]) # top distances
    # counter
    c = 0
    for i in range(len(yes)):
        # if value in row is nan
        if yes[i] != yes[i]:
            # get value of column from rows with closest cosine distance
            R = np.array([int(i) for i in miss[top_indexes]])
            
            # calculate weighted average where 1-distance is the weight
            num = sum((1-D)*R)
            den = k - sum(D)
            
            # to round, or not to round
            if dec != None:
                yes[i] = round(num/den,dec)
            else:
                yes[i] = num/den
            c+=1
    # return complete row
    return yes
    		
def wa_imputer(df,miss=np.nan,k=2,dec=None):
    # ensures pd.dataframe
    df = pd.DataFrame(df)
    # replace missing values with nan for consistency
    df = df.replace(miss,np.nan)
    
    # contains rows with missing values
    no_na = df.dropna()
    
    # constains rows without missing values
    yes_na = df[~df.index.isin(no_na.index)]

    for row in yes_na.iterrows():   # row[0] is row index id, row[1] is values of row
        
        # the columns of values that are not missing (same column as the row's nan values)
        miss = np.array(no_na[no_na.columns[np.isnan(row[1])]])
        
        # obtain filled in row
        new_row = weightedAverage(row[1],no_na,miss,k,dec)
        
        # replace row
        df.loc[row[0]] = new_row
    return df
    
	