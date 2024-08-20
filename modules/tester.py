import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm

from modules.utils import generate_heatmap


class BaseTester(object):
    def __init__(self, model, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        # self.model.load_state_dict(checkpoint['state_dict'])

        state_dict = checkpoint['state_dict']
        try:
            self.model.load_state_dict(state_dict)
        except:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)


class Tester(BaseTester):
    def __init__(self, model, metric_ftns, args, test_image):
        super(Tester, self).__init__(model, metric_ftns, args)
        self.test_image = test_image
    def test(self):
        self.model.eval()
        with torch.no_grad():
            images = self.test_image.to(self.device)
            output= self.model(images, mode='sample')
            # output, _ = self.model(images, mode='sample')
            reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        return reports

