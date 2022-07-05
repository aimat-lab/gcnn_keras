from sklearn.preprocessing import StandardScaler as StandardScalerSklearn


class StandardScaler(StandardScalerSklearn):

    def __init__(self, **kwargs):
        super(StandardScaler, self).__init__(**kwargs)

    def get_config(self):
        config = {"copy": self.copy, "with_mean": self.with_mean, "with_std": self.with_std}
        return config

    def get_weights(self):
        pass

    def set_weights(self):
        pass

