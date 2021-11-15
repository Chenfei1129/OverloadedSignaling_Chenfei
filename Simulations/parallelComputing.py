  
#Parallel Computation
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
def parallelize_dataframe(df, func, n_cores=2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis = 1)

def parallelize_on_row(df, func, ncore=2):
    return parallelize_dataframe(df, partial(run_on_subset, func), ncore)
