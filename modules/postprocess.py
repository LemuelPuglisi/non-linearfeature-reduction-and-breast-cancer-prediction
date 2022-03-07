import numpy  as np 
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import model_from_json
import seaborn as sns

def plot_history(hist):  
    """ Plot loss during training process """
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def visualize_reconstruction(original_ds, reconstructed_ds, _range = range(0,3)):
    """ Show input and output molecular sequence """
    osam = original_ds.iloc[_range]
    rsam = reconstructed_ds[_range]
    feat = len(rsam[0])
    for i in _range:
        print(f"Sample n{i+1}")
        print('-' * 50)
        for j in range(0, feat):
            ori = osam.iloc[i, j]
            pre = rsam[i][j]
            out = "original {:.2f} \t| {:.2f} predicted".format(ori, pre)    
            print(out)
        v1 = osam.iloc[i].to_numpy()
        v2 = rsam[i]
        print(">> square error: {:.6f}".format(__square_error(v1, v2)))
        print("\n\n")  

      
def __square_error(v1, v2):
    return ((v1 - v2) ** 2).sum()  
        
def visualize_as_images(dataframe, autoencoder, index, colorbar = False, cm = 'viridis'):
    """ Show input and output molecular sequence as mosaic images"""
    inseq = dataframe.iloc[index]
    inseq_asdf = inseq.to_frame().transpose()
    prarr = autoencoder.predict(inseq_asdf)
    inarr = inseq.to_numpy()
    arlen = len(inarr)
    
    edge_len = np.ceil(np.sqrt(arlen)).astype('int')
    img_size = edge_len ** 2
    
    if (arlen < img_size):
        buffer_in = np.zeros(img_size)
        buffer_pr = np.zeros(img_size)
        buffer_in[:arlen] = inarr
        buffer_pr[:arlen] = prarr
        inarr = buffer_in
        prarr = buffer_pr
        
    im_inp = np.reshape(inarr, (edge_len, edge_len))
    im_pre = np.reshape(prarr, (edge_len, edge_len))

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(im_inp, cmap=cm, vmin=0, vmax=1)
    if (colorbar):
        plt.colorbar()
    f.add_subplot(1, 2, 2)
    plt.imshow(im_pre, cmap=cm, vmin=0, vmax=1)
    if (colorbar):
        plt.colorbar()
    plt.show()
    
    
def save_model(model, name, directory = "models"):
    """ Save the model to the specified directory """
    model_json = model.to_json()
    mod_dest = directory + '/' + name + '.json'
    wgt_dest = directory + '/' + name + '_weights.h5'
    with open(mod_dest, "w") as json_file:
        json_file.write(model_json)
        model.save_weights(wgt_dest)
        

def load_model(model_path, weights_path):
    """ Load a model from the given path """
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model


def evaluate_and_display(model, validation_set):
    """ Evaluate the model and prints the model evaluation metrics """
    metrics_names = model.metrics_name
    metrics_values = model.evaluate(validation_set, validation_set, verbose=0)
    for idx, val in enumerate(metrics):
        print(f'{value}:\t {metrics_values[idx]}')
        

def save_compressed_dataset(dataframe, loss, savepath, filename):
    """
    Saves the compressed dataset into a directory, specified by the savepath. 
    """
    parts = filename.split('/')
    directory = parts[0]
    filename  = parts[1] 
    embedding_space = dataframe.shape[1]    
    path = f'{savepath}/loss_{loss:.4f}_{embedding_space}f_reduced_{filename}'
    dataframe.to_csv(path)
    

# def save_compressed_dataset(ds, loss, features, csv_path, results_dir = 'results'):
# """
# Takes the target directory (ngs, microarray) directly from the csv_path
# and build a new path to store the compressed dataset into the results folder.
# """
# parts = csv_path.split('/')
# directory = parts[0]
# filename  = parts[1] 
# save_path = f'{results_dir}/{directory}/loss_{loss:.4f}_{features}f_reduced_{filename}'
# ds.to_csv(save_path)
    

def produce_binary_cm (metrics, dimensions, classes, dataset_name, plots_directory = 'plots'): 
    """
    For each embedding space (dimensions) produce a plot using matplotlib and seaborn.
    For each subtype of tumor, it produces a binary confusion matrix using a one vs all 
    approach, and it calculates some metrics to display aside the confusion matrix. 
    Dimensions parameter contains a list of embedding spaces (i.e. [150, 50, 25]). 
    Classes parameter contains a list of tumor subtypes (i.e. ['Her2', 'LumA']).
    Metrics parameter is an object that contains a key for each embedding space; the 
    corresponding value is another object, containing accuracy score obtained using 
    the reduced dataset, and a bunch of label specific metrics (sensitivity, npv, ...).
    """
    for embedding_space in dimensions: 
        pidx = 1 
        cols = 2
        rows = len(classes)

        fig, ax = plt.subplots(rows, cols, figsize=(15,20))
        fig.suptitle("Binary confusion matrices (one vs all)", fontsize=16)

        current_metrics = metrics.get(embedding_space)
        label_specific_metrics = current_metrics.get('label_specific')

        for subtype in classes: 

            subtype_metrics = label_specific_metrics.get(subtype)
            confusion_matrix = subtype_metrics.get('confusion_matrix')
            
            subtype_name = subtype
            if '_' in subtype:
                subtype_name = subtype.split('_')[1]

            # plot the c. matrix as an heatmap
            plt.subplot(rows, cols, pidx)
            title = plt.title(f'{subtype_name} (dim. {embedding_space})')
            xlabels = ['0 (pred)', '1 (pred)']
            ylabels = ['0 (real)', '1 (real)']
            sns.heatmap(confusion_matrix, 
                        xticklabels=xlabels, 
                        yticklabels=ylabels, 
                        annot=True, cbar=False, fmt='g')
            pidx += 1

            # plot the scores aside the confusion matrix
            data = [['sensitivity', subtype_metrics.get('sensitivity')],
                    ['specificity', subtype_metrics.get('specificity')],
                    ['precision',   subtype_metrics.get('precision')],
                    ['npv',         subtype_metrics.get('npv')]]

            scores = pd.DataFrame(data, columns=['metric', 'score'])
            plt.subplot(rows, cols, pidx)
            g = sns.barplot(x="metric", y="score", data=scores)
            
            # add a threshold line to 1 
            plt.axhline(y=1,linewidth=0.75, color='green', linestyle='--')

            # some magic to display the values of the barchart
            ax=g
            for p in ax.patches:
                ax.annotate(
                    "%.2f" % p.get_height(), 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, 
                    color='black', xytext=(0, 20), textcoords='offset points')
            _ = g.set_ylim(0,1.2) # to make space for the annotations

            pidx+=1
        
        figname = f'{plots_directory}/binary_confusion_matrix_{dataset_name}_f{embedding_space}.png'   
        plt.savefig(figname, dpi=150, transparent=False)