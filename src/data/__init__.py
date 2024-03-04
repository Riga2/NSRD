from importlib import import_module
from data import data_loader
import os

class Data:
    def __init__(self, args):
        self.sr_content = args.sr_content
        self.loader_train = None
        module_name = self.sr_content
        m = import_module('data.' + module_name.lower() + '_dataset')
        if not args.test_only:
            train_datasets = m.make_model(args, train=True)
            valid_datasets = m.make_model(args, train=False)
            self.loader_train = data_loader.RenderingDataLoader(args, train_datasets)
            self.loader_valid = self.loader_train.split_validation(valid_datasets)
        else:
            test_datasets = m.make_model(args, train=False)
            self.loader_valid = data_loader.RenderingDataLoader(args, test_datasets)

