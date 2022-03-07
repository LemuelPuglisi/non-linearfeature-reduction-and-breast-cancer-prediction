import pandas as pd
from modules.preprocess import adapt, reduce_columns

def get_data_sources():
    return [
        'microarray/data_mRNA_median_Zscores.txt', 
        'microarray/data_mRNA_median_all_sample_Zscores.txt', 
    ]

def get_ds(csv_index=0, ds_dir='datasets', delete_const=True): 
    """ Return a preprocessed dataset """
    ds_list = get_data_sources()
    ds = pd.read_csv(f'{ds_dir}/{ds_list[csv_index]}', sep='\t')
    return adapt(ds, 'Hugo_Symbol', delete_const)


def get_clinical_ds():
    return pd.read_csv('datasets/microarray/data_clinical_patient.txt', skiprows=4, sep='\t')


def find_patient_clinical_data(sample_id, clinical_features, clinical_ds):
    # in this case patient_id and sample_id are the same 
    bio = clinical_ds[clinical_ds['PATIENT_ID'] == sample_id].iloc[0]
    return bio[clinical_features]


def get_samples_with_label(label):
    samples_ds  = get_ds()
    clinical_ds = get_clinical_ds()
    
    # extract the label
    samples_ids = samples_ds.index.to_numpy()
    subtypes_ds = [ find_patient_clinical_data(sid, [label], clinical_ds) for sid in samples_ids]
    subtypes_ds = pd.DataFrame(subtypes_ds, columns=[label])
    subtypes_ds.index = samples_ds.index
    
    # concat to the genes dataset 
    df = pd.concat( [ samples_ds, subtypes_ds[label] ], axis=1)
    rows_without_label = df[label].isnull()
    return df[-rows_without_label]

    return df[-rows_without_label]


def attach_label(samples_ds, label):
    clinical_ds = get_clinical_ds()
    
    # extract the label
    samples_ids = samples_ds.index.to_numpy()
    subtypes_ds = [ find_patient_clinical_data(sid, [label], clinical_ds) for sid in samples_ids]
    subtypes_ds = pd.DataFrame(subtypes_ds, columns=[label])
    subtypes_ds.index = samples_ds.index
    
    # concat to the genes dataset 
    df = pd.concat( [ samples_ds, subtypes_ds[label] ], axis=1)
    rows_without_label = df[label].isnull()
    return df[-rows_without_label]