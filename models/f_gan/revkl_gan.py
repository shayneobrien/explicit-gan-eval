import models.f_gan.f_gan as f_gan
from models.f_gan.f_gan import *


class Trainer(f_gan.Trainer):
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, train_iter, val_iter, test_iter,
                    method='reverse_kl', viz=False):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.method = method
        self.viz = viz
        self.metrics = defaultdict(list)

        self.As = []
        self.Bs = []
