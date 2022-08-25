from hashlib import md5
import logging

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

module_logger.warning(
    "Modules in `kgcnn.crystal` are still in development and not fully tested.")


class CrystalPreprocessor:

    def __call__(self, structure) -> dict:
        raise NotImplementedError()

    def get_config(self):
        config = vars(self)
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        config['preprocessor'] = self.__class__.__name__
        return config

    def hash(self):
        return md5(str(self.get_config()).encode()).hexdigest()

    def __hash__(self):
        return int(self.hash(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)
