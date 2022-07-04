from sklearn.preprocessing import StandardScaler as StandardScalerSklearn


class StandardScaler(StandardScalerSklearn):

    def __init__(self, **kwargs):
        super(StandardScaler, self).__init__(**kwargs)

    def get_config(self):
        pass

    def set_config(self):
        pass

    def from_config(self):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass

