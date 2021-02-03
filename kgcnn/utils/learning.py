import numpy as np



def lr_lin_reduction(learning_rate_start = 1e-3,learning_rate_stop = 1e-5,epo = 10000,epomin= 1000):
    """
    Make learning rate schedule function for linear reduction.

    Args:
        learning_rate_start (float, optional): Learning rate to start with. The default is 1e-3.
        learning_rate_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epo (int, optional): Total number of epochs to reduce learning rate towards. The default is 10000.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 1000.

    Returns:
        func: Function to use with LearningRateScheduler.
    
    Example:
        lr_schedule_lin = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction)
    """
    def lr_out_lin(epoch):
        if(epoch < epomin):
            out = learning_rate_start
        else:
            out = float(learning_rate_start - (learning_rate_start-learning_rate_stop)/(epo-epomin)*(epoch-epomin))
        return out
    return lr_out_lin