import pandas as pd 
import numpy  as np 


def reduce_columns(ds, target_dim):
    """ Reduce the number of columns to target_dim """
    return ds.drop(ds.columns[target_dim:], axis=1)


def adapt(ds, colname, delete_constants=True):
    """ Apply preprocessing to the dataset """
    colnames = ds[colname]
    col_to_drop = ['Hugo_Symbol', 'Entrez_Gene_Id']
    tmp = ds.transpose().fillna(0).drop(col_to_drop)
    tmp.columns = colnames
    if (delete_constants):
        tmp = delete_constant_cols(tmp)
    return tmp


def delete_constant_cols(ds):
    """ Delete constant columns in a dataframe """
    constant_cols = find_constant_cols(ds)
    return ds.drop(constant_cols, axis=1)


def is_constant(arr):
    """ Return true if the array contains const. values """
    res = arr[0] == arr
    return res.all()


def find_constant_cols(ds):
    """ Return a list of the constant columns in a dataframe """
    constant_cols = []
    for col in ds.columns:
        if (is_constant(ds[col].to_numpy())):
            constant_cols.append(col)
    return constant_cols