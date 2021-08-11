from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, CuDNNGRU, CuDNNLSTM, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
          
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    #print(model.output_length)
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
  

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_rnn_stack_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    ###################################
    #add a second convolution layer
    conv_1d_2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_2')(bn_cnn)
    # Add batch normalization
    bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
    ##################################
    
    
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn_2)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    
    GRU / LSTM IO ‘CuDNNGRU / CuDNNLSTM
    """
    
    activation ='relu'
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    
    # Add recurrent layer #1 ----------------------------------
    rnn_1 = CuDNNGRU(units, return_sequences=True, name='rnn_1')(input_data)
    # Add batch normalization for layer #1
    bn_rnn_1 = BatchNormalization(axis=-1)(rnn_1)
    
      # Add recurrent layer #2 ----------------------------------
    rnn_2 = CuDNNGRU(units,return_sequences=True,name='rnn-2')(bn_rnn_1)
    # Add batch normalization for layer #1
    bn_rnn_2 = BatchNormalization(axis=-1)(rnn_2)
    
    ########################################################################
          # Add recurrent layer #3 : training was too slow, and performance did not sensibly increase----------------
    #rnn_3 = GRU(units, activation=activation,
        #return_sequences=True, implementation=2, name='rnn_3')(bn_rnn_2)
    # Add batch normalization for layer #1
    #bn_rnn_3 = BatchNormalization(axis=-1)(rnn_3)
    #########################################################################
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer##########
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_2)
    
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def abyss_deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    
    GRU / LSTM IO ‘CuDNNGRU / CuDNNLSTM
    """
    
    activation ='relu'
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn_0 = BatchNormalization(axis=-1)(input_data)


    # Add recurrent layer #1 ----------------------------------
    #rnn_1 = CuDNNGRU(units, return_sequences=True, name='rnn_1')(input_data)
    rnn_1 = CuDNNGRU(units, return_sequences=True, name='rnn_1')(bn_rnn_0)
    # Add batch normalization for layer #1
    bn_rnn_1 = BatchNormalization(axis=-1)(rnn_1)
    
      # Add recurrent layer #2 ----------------------------------
    rnn_2 = CuDNNGRU(units,return_sequences=True,name='rnn-2')(bn_rnn_1)
    # Add batch normalization for layer #1
    bn_rnn_2 = BatchNormalization(axis=-1)(rnn_2)
    
    ########################################################################
    # Add recurrent layer #3 : training was too slow, and performance did not sensibly increase
    rnn_3 = CuDNNGRU(units,return_sequences=True,name='rnn-3')(bn_rnn_2)
    # Add batch normalization for layer #1
    bn_rnn_3 = BatchNormalization(axis=-1)(rnn_3)
    
    #########################################################################
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer##########
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_3)
    
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


#def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = ...
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
###########################



def bidirectional_rnn_model(input_dim, units, output_dim=29):
   
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


#https://programtalk.com/python-examples/keras.layers.MaxPooling1D/

def cnn_funnel_rnn_model_2_layers(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # ADD 1ST CONVOLUTION 
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d1')(input_data)
    
  
        # Add batch normalization   
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_1 = Dropout(dropout_rate)(bn_cnn_1)
    
    # ADD 2ND CONVOLUTION 

    conv_1d_2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d2')(bn_cnn_1)
    
                ##########conv_2d = MaxPooling1D(2)(bn_cnn_1)
        
        # Add batch normalization   
    bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_2 = Dropout(dropout_rate)(bn_cnn_2)
    
    
    # Add RNN
    
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn_2)
    
    # over fitting Add drop out
    simp_rnn = Dropout(dropout_rate)(simp_rnn)
    
    # Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_funnel_rnn_model_MAX_POOL(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layer
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d1')(input_data)
    
    #############
    conv_2d = MaxPooling1D(2)(conv_1d_1)
    ############    
        # Add batch normalization   
    #bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
    #############
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_2d)


        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_1 = Dropout(dropout_rate)(bn_cnn_1)
    
    # Add convolutional layer
    #conv_1d_2 = Conv1D(filters, kernel_size, 
                     #strides=conv_stride, 
                     #padding=conv_border_mode,
                     #activation='relu',
                     #name='conv1d2')(bn_cnn_1)
    
                ##########conv_2d = MaxPooling1D(2)(bn_cnn_1)
        
        # Add batch normalization   
    #bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
        # overfitting->Add drop out
    dropout_rate=0.2
    #bn_cnn_2 = Dropout(dropout_rate)(bn_cnn_2)
    
    
    # Add RNN
    
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn_1)
    
    # over fitting Add drop out
    simp_rnn = Dropout(dropout_rate)(simp_rnn)
    
    # Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_drop_out_deep_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
# Add convolutional layer

    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d1')(input_data)
    
    # Add batch normalization   
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)    
        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_1 = Dropout(dropout_rate)(bn_cnn_1)
##########################################################################################################    

# Add recurrent layer #1 ----------------------------------
    rnn_1 = CuDNNGRU(units, return_sequences=True, name='rnn_1')(bn_cnn_1)
    
    # Add batch normalization for layer #1
    bn_rnn_1 = BatchNormalization(axis=-1)(rnn_1)
    
# Add recurrent layer #2 ----------------------------------

    rnn_2 = CuDNNGRU(units,return_sequences=True,name='rnn-2')(bn_rnn_1)
    # Add batch normalization for layer #1
    bn_rnn_2 = BatchNormalization(axis=-1)(rnn_2)
    
# Add recurrent layer #3 : training was too slow, and performance did not sensibly increase
    rnn_3 = CuDNNGRU(units,return_sequences=True,name='rnn-3')(bn_rnn_2)
    # Add batch normalization for layer #1
    bn_rnn_3 = BatchNormalization(axis=-1)(rnn_3)
    
    
    
#  Add a TimeDistributed(Dense(output_dim)) layer##########
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_3)
       
#########################################################################################################  
    
# Add softmax activation layer

    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
##########



def cnn_drop_out_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
# Add convolutional layer

    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d1')(input_data)
    
    # Add batch normalization   
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)    
        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_1 = Dropout(dropout_rate)(bn_cnn_1)
    
# Add RNN
    
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn_1)
    
    # over fitting Add drop out
    simp_rnn = Dropout(dropout_rate)(simp_rnn)
    
    # Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
# Add softmax activation layer

    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
##########
"""
model_5 = cnn_funnel_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)

"""
#def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):


###############
# ADDING BIAS?
# REDUCE  THE SIZE OF THE KERNEL THROUGH THE CNN STACK
# SHOULD DROPPUT RATE VARY THROUGOUT THE STACK 
# LEARNING RATE

#rnn_input = Dropout(dropout_rate)(rnn_input)
#MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
"""
rnn = Bidirectional(GRU(128, return_sequences = True))(input_seq)

CuDNNLSTM
go_backwards=False
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
def rnn_model(input_dim, units, activation, output_dim=29):
     Build a recurrent network for speech 
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
  

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
"""


def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
# Add convolutional layer

    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d1')(input_data)
    
    # Add batch normalization   
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)    
        # overfitting->Add drop out
    dropout_rate=0.2
    bn_cnn_1 = Dropout(dropout_rate)(bn_cnn_1)
##########################################################################################################    

# Add recurrent layer #1 ----------------------------------
    rnn_1 = CuDNNGRU(units, return_sequences=True, name='rnn_1')(bn_cnn_1)
    
    # Add batch normalization for layer #1
    bn_rnn_1 = BatchNormalization(axis=-1)(rnn_1)
    
# Add recurrent layer #2 ----------------------------------

    rnn_2 = CuDNNGRU(units,return_sequences=True,name='rnn-2')(bn_rnn_1)
    # Add batch normalization for layer #1
    bn_rnn_2 = BatchNormalization(axis=-1)(rnn_2)
    
# Add recurrent layer #3 : training was too slow, and performance did not sensibly increase
    rnn_3 = CuDNNGRU(units,return_sequences=True,name='rnn-3')(bn_rnn_2)
    # Add batch normalization for layer #1
    bn_rnn_3 = BatchNormalization(axis=-1)(rnn_3)
    
    
    
#  Add a TimeDistributed(Dense(output_dim)) layer##########
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_3)
       
#########################################################################################################  
    
# Add softmax activation layer

    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
##########