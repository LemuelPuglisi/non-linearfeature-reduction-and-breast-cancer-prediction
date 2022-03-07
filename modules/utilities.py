import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


def train_test_split(df, test_perc):
    """ 
    Returns 2 dataframes, a training set containing 
    (100 - test_perc * 100) percent of the dataset 
    and a test set containing (test_perc * 100) percent
    of the dataset. 
    """
    if (test_perc < 0 or test_perc > 1):
        return None, None
    nx, _ = df.shape
    breakpoint = int((1 - test_perc) * nx)
    return df[:breakpoint], df[breakpoint:]


def multilabel_train_test_split(ds, label_column, test_perc = .1):
    """ 
    Returns 2 dataframes, a training set containing 
    (100 - test_perc * 100) percent of the dataset 
    and a test set containing (test_perc * 100) percent
    of the dataset. This method ensures that every label
    is contained in each set. 
    """
    if (test_perc < 0 or test_perc > 1):
        return None, None
    
    labels = ds[label_column].unique()
    n, _ = ds.shape 
    
    trn_subset = pd.DataFrame(data=None, columns=ds.columns)
    tst_subset = pd.DataFrame(data=None, columns=ds.columns)
    
    for label in labels:
        
        label_class_data = ds[ds[label_column] == label]
        label_class_rows = len(label_class_data)
        test_subset_rows = round((label_class_rows / n) * (n * test_perc)) 
        
        tst_subset_to_add = label_class_data.iloc[:test_subset_rows] 
        trn_subset_to_add = label_class_data.iloc[test_subset_rows:] 
        tst_subset = pd.concat([tst_subset, tst_subset_to_add])
        trn_subset = pd.concat([trn_subset, trn_subset_to_add])
            
    return shuffle(trn_subset), shuffle(tst_subset)


def normalize_sets(train_df, test_df, scaler = MinMaxScaler()):
    """ Normalize train and test set between 0 and 1 """
    # saving indexes and column names  
    # before normalization
    tr_colnames = train_df.columns
    tr_rownames = train_df.index 
    te_colnames = test_df.columns
    te_rownames = test_df.index
    # normalize
    scaler.fit(X=train_df.to_numpy())
    output_train_df = pd.DataFrame(scaler.transform(train_df.to_numpy()))
    output_test_df  = pd.DataFrame(scaler.transform(test_df.to_numpy()))
    # restoring the prev  columns names
    # and indexes 
    output_train_df.columns = tr_colnames
    output_train_df.index = tr_rownames
    output_test_df.columns = te_colnames
    output_test_df.index = te_rownames
    # return sets
    return output_train_df, output_test_df, scaler


def normalize_with_pretrained_scaler(scaler, dataset):
    colnames = dataset.columns
    rownames = dataset.index 
    norm_dataset = pd.DataFrame(scaler.transform(dataset.to_numpy()))
    norm_dataset.columns = colnames
    norm_dataset.index = rownames
    return norm_dataset