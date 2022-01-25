import abc

class TrainingMethod(metaclass=abc.ABCMeta):
   @abc.abstractmethod
   def train_epoch(self, input_batch, target_batch):
      pass

   @abc.abstractmethod
   def val_epoch(self, input):
      pass