import numpy as np

from numpy import linalg as la
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.losses import Huber
from keras.constraints import maxnorm


def generate_autoencoder(input_dim, 
                         latent_space, 
                         encoder_activation='relu',
                         decoder_activation='sigmoid', 
                         optimizer='adam',
                         loss_function=None):
    
    """ Generate a simple autoencoder for dimensionality reductions. 
        Both the encoder and decoder have 1 layer. The input_dim is the 
        dimension of the input, which will be compressed into the latent 
        space, hence a space with latent_space dimensions. 
    """
    encoder_in = Input(shape=(input_dim, ))
    encoder = Dense(latent_space, activation=encoder_activation)(encoder_in)
    decoder = Dense(input_dim, activation=decoder_activation)(encoder)
    if (loss_function is None):
        loss_function = Huber(delta=1.0, reduction="auto", name="huber_loss")
    autoencoder = Model(encoder_in, decoder)
    autoencoder.compile(optimizer=optimizer, loss=loss_function)
    return autoencoder




def generate_deep_autoencoder(input_dim, encoder_components, decoder_components, options = {}):
    
    """ This function generate a deep autoencoder. The layers for both encoder
        and decoder are passed through 2 lists. The first list, encoder_components, 
        contains dictionaries; each dictionary contains 2 attributes: "latent_space", 
        which is the number of units, and "activation" which indicates the activation
        function used in the layer. The same structure is defined for the decoder. 
        Example:

        encoder_component = [
            {'latent_space': 7, 'activation': 'relu', 'batch_norm_output': 'true'}, 
            {'latent_space': 4, 'activation': 'relu', 'batch_norm_output': 'true'}
        ]

        decoder_component = [
            {'latent_space': 7,  'activation': 'relu',    'batch_norm_output': 'false'}, 
            {'latent_space': 10, 'activation': 'sigmoid', 'batch_norm_output': 'false'}
        ]
    """
    optimizer = options.get('optimizer', 'adam')
    loss_func = options.get('loss_function', Huber(delta=1.0, reduction="auto", name="huber_loss"))
    dropout   = options.get('dropout', True)
    metrics   = options.get('metrics', ['mae', 'mse'])
    # learning_rate = options.get('learning_rate', 10)
    # momentum = options.get('momentum', .9)

    autoenc = inputs = Input(shape=(input_dim, ))
    firstcomp = encoder_components[0]
    if (dropout):
        autoenc = Dropout(rate=.2)(autoenc)
    autoenc = __add_dense_layer(autoenc, firstcomp, dropout)

    for ecomp in encoder_components[1:]:
        autoenc = __add_dense_layer(autoenc, ecomp, dropout)
    for dcomp in decoder_components[0:-1]:
        autoenc = __add_dense_layer(autoenc, dcomp, dropout)
    
    lastcomp = decoder_components[-1]
    lastcomp['batch_norm_output'] = False # just to make sure
    autoenc = __add_dense_layer(autoenc, lastcomp, False)

    autoencoder = Model(inputs, autoenc)
    autoencoder.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return autoencoder  


def generate_biased_autoencoder(input_dim, encoder_components, classifier_components, options = {}):
    
    optimizer = options.get('optimizer', 'adam')
    loss_func = options.get('loss_function', 'categorical_crossentropy')
    dropout   = options.get('dropout', True)
    metrics   = options.get('metrics', ['accuracy'])
    
    # creating the encoder component of the autoencoder as usually
    autoenc = inputs = Input(shape=(input_dim, ))
    firstcomp = encoder_components[0]
    if (dropout):
        autoenc = Dropout(rate=.2)(autoenc)
    autoenc = __add_dense_layer(autoenc, firstcomp, dropout)
    for ecomp in encoder_components[1:]:
        autoenc = __add_dense_layer(autoenc, ecomp, dropout)
    # replacing the decoder part with a classifier 
    for ccomp in classifier_components[0:-1]:
        autoenc = __add_dense_layer(autoenc, ccomp, dropout)
    
    lastcomp = classifier_components[-1]
    lastcomp['batch_norm_output'] = False # just to make sure
    autoenc = __add_dense_layer(autoenc, lastcomp, False)

    autoencoder = Model(inputs, autoenc)
    autoencoder.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return autoencoder  
    
    
def __add_dense_layer(model, comp, dropout):
    if (dropout):
        ## as specified in the dropout paper, the max value of the weight norm must be under 3 
        model = Dense(comp['latent_space'], activation=comp['activation'], kernel_constraint=maxnorm(3))(model)
    else:
        model = Dense(comp['latent_space'], activation=comp['activation'])(model)
    if (comp['batch_norm_output']):
        model = BatchNormalization(axis=1)(model)
    return model


def extract_encoder(autoencoder, encoder_layer_index):
    """
    Extract the encoder component from an autoencoder passing 
    the full trained autoencoder and the index of the last encoder
    layer. 
    """
    return Model(inputs=autoencoder.input, 
                 outputs=autoencoder.layers[encoder_layer_index].output)


def blind_autoencoders(ninput):
    return {
        150: {
            'encoder': [
                {'latent_space': 15000, 'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 7500,  'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 3000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 1000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 150,   'activation': 'relu', 'batch_norm_output': True},
            ], 
            'decoder': [
                {'latent_space': 1000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 3000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 7500,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 15000, 'activation': 'relu',    'batch_norm_output': False}, 
                {'latent_space': ninput,'activation': 'sigmoid', 'batch_norm_output': False}
            ], 
            'encoder_position': 10
        }, 
        50: {
            'encoder': [
                {'latent_space': 15000, 'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 7500,  'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 3000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 1000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 150,   'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 50,    'activation': 'relu', 'batch_norm_output': True},
            ], 
            'decoder': [
                {'latent_space': 150,   'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 1000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 3000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 7500,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 15000, 'activation': 'relu',    'batch_norm_output': False}, 
                {'latent_space': ninput,'activation': 'sigmoid', 'batch_norm_output': False}
            ], 
            'encoder_position': 13
        }, 
        25: {
            'encoder': [
                {'latent_space': 15000, 'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 7500,  'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 3000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 1000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 150,   'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 25,    'activation': 'relu', 'batch_norm_output': True},
            ], 
            'decoder': [
                {'latent_space': 150,   'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 1000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 3000,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 7500,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 15000, 'activation': 'relu',    'batch_norm_output': False}, 
                {'latent_space': ninput,'activation': 'sigmoid', 'batch_norm_output': False}
            ], 
            'encoder_position': 13
        }
    }


def biased_autoencoder(nclasses):
    return {
        'SubBIA': {
            'encoder': [
                {'latent_space': 15000, 'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 7500,  'activation': 'relu', 'batch_norm_output': True}, 
                {'latent_space': 3000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 1000,  'activation': 'relu', 'batch_norm_output': True},
                {'latent_space': 150,   'activation': 'relu', 'batch_norm_output': True},
            ], 
            'classifier': [
                {'latent_space': 100,      'activation': 'relu',    'batch_norm_output': True},
                {'latent_space': 50,       'activation': 'relu',    'batch_norm_output': True},
                {'latent_space': 25,       'activation': 'relu',    'batch_norm_output': True},
                {'latent_space': nclasses, 'activation': 'softmax', 'batch_norm_output': False}
            ], 
            'encoder_position': {
                150: 11, 
                50 : 15,
                25 : 17
            }
        }, 
        'SurBIA': {
            'encoder': [
                {'latent_space': 15000, 'activation': 'relu', 'batch_norm_output': False}, 
                {'latent_space': 7500,  'activation': 'relu', 'batch_norm_output': False}, 
                {'latent_space': 3000,  'activation': 'relu', 'batch_norm_output': False},
                {'latent_space': 1000,  'activation': 'relu', 'batch_norm_output': False},
                {'latent_space': 150,   'activation': 'relu', 'batch_norm_output': False},
            ], 
            'regressor': [
                {'latent_space': 128,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 128,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 128,  'activation': 'relu',    'batch_norm_output': False},
                {'latent_space': 1,    'activation': None, 'batch_norm_output': False}
            ]
        }
        
    }
   


def compute_neurons(n, a, b): 
    """ Given n layers, input size "a" and embedding size "b", this function
        solves a system of linear equations to return the distribution of nodes
        into the n layers, supposing we need to split the number of nodes up 
        every next layer (credits to prof. Alaimo Salvatore). 
    """
    if (n < 1):
        raise Exception("Invalid number of levels")
    if (n == 1): 
        return ((a + b) // 2)
    
    # matrix of coefficients
    M = np.zeros((n,n))
    M[0, [0,1]] = [2, -1]
    M[n-1, [n-2, n-1]] = [-1, 2]
    if ((n-1) >= 2):
        for i in range(1, n-1):
            M[i, [i-1, i, i+1]] = [-1, 2, -1]

    # intercepts 
    bterms = np.zeros((n, ))
    bterms[0] = a
    bterms[n-1] = b

    # solve the linear system 
    return np.floor(la.solve(M, bterms))


def generate_alaimo_autoencoder(input_size, embedding_size): 

    units = compute_neurons(3, input_size, embedding_size)
    inputs = Input(shape=(input_size, ))

    # encoder
    encoder = Dense(units[0], activation='tanh')(inputs)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = Dropout(rate=0.25)(encoder)
    encoder = Dense(units[1], activation='tanh')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = Dropout(rate=0.25)(encoder)
    encoder = Dense(units[2], activation='tanh')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = Dropout(rate=0.25)(encoder)
    encoder = Dense(embedding_size, activation='tanh')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = Dropout(rate=0.25)(encoder)

    # decoder
    decoder = Dense(units[2], activation='tanh')(encoder)
    decoder = BatchNormalization(axis=1)(decoder)
    decoder = Dropout(rate=0.25)(decoder)
    decoder = Dense(units[1], activation='tanh')(decoder)
    decoder = BatchNormalization(axis=1)(decoder)
    decoder = Dropout(rate=0.25)(decoder)
    decoder = Dense(units[0], activation='tanh')(decoder)
    decoder = BatchNormalization(axis=1)(decoder)
    decoder = Dropout(rate=0.25)(decoder)
    decoder = Dense(input_size)(decoder)
    decoder = BatchNormalization(axis=1)(decoder)
    decoder = Dropout(rate=0.25)(decoder)

    autoencoder = Model(inputs, decoder)
    autoencoder.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics='mean_absolute_error')
    return autoencoder  
