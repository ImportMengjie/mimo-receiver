import abc


class Tee(abc.ABC):

    def __init__(self, items):
        self.items = items

    @abc.abstractmethod
    def get_model_input(self):
        pass

    @abc.abstractmethod
    def set_model_output(self, outputs):
        pass

    @abc.abstractmethod
    def get_loss_input(self):
        pass
