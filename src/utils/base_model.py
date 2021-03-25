from abc import abstractmethod, ABC


class BaseModel(ABC):

    def __init__(self, *args, **kwargs):
        self.model = self.model(*args, **kwargs)

    @abstractmethod
    def model(self, *args, **kwargs):
        raise NotImplementedError

    def disable(self):
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

    def enable(self):
        for layer in self.model.layers:
            layer.trainable = True
        self.model.trainable = True
