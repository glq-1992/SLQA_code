from __future__ import print_function
import os
from collections import OrderedDict
import torch


class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model_state during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        model_state (seq2seq): seq2seq_model model_state being trained
        optimizer_state (Optimizer): stores the state of the optimizer_state
        epoch (int): current epoch (an epoch is a loop through the full training data)
        epoch (int): number of examples seen within the current epoch

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_STATE_NAME (str): name of the file storing model_state
    """

    CHECKPOINT_DIR_NAME = 'checkpoint'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_STATE_NAME = 'model_states.pt'

    def __init__(self, model_state, optimizer_state, epoch, min_er, min_cost, path=None):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.epoch = epoch
        self.min_er = min_er
        self.min_cost = min_cost
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):
        """
        Saves the current model_state and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """

        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME)
        path = self._path

        os.makedirs(path, exist_ok=True)
        torch.save({'epoch': self.epoch,
                    'min_er': self.min_er,
                    'min_cost': self.min_cost,
                    'optimizer_state': self.optimizer_state
                    }, os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model_state, os.path.join(path, self.MODEL_STATE_NAME))

        return path

    @classmethod
    def load(cls, path, multi2single=False):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model_state = torch.load(os.path.join(path, cls.MODEL_STATE_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),
                                           map_location=lambda storage, loc: storage)
            model_state = torch.load(os.path.join(path, cls.MODEL_STATE_NAME),
                                     map_location=lambda storage, loc: storage)
        if multi2single:
            new_model_state = OrderedDict()
            for k, v, in model_state.items():
                namekey = k[7:]
                new_model_state[namekey] = v
            model_state = new_model_state

        optimizer_state = resume_checkpoint['optimizer_state']
        return Checkpoint(model_state=model_state,
                          optimizer_state=optimizer_state,
                          epoch=resume_checkpoint['epoch'],
                          min_er=resume_checkpoint['min_er'],
                          min_cost=resume_checkpoint['min_cost'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, "min_er", cls.CHECKPOINT_DIR_NAME)
        return checkpoints_path
