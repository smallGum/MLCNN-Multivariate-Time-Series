"""
    Implements optimization algorithms.
"""

from utils._libs_ import math, optim

# -------------------------------------------------------------------------------------------------------------------------------------------------
class Optimize():
    def __init__(self, params, method, lr, max_grad_norm, lr_decay=0.1, start_decay_at=None, patience=5, decay_nb=3):
        """
        Initialization arguments:
            params          - (torch.nn.Module.parameters())  parameters of model (may be a generator)
            method          - (string)                        optimization method
            lr              - (float)                         learning rate
            max_grad_norm   - (float)                         norm for gradient clipping
            lr_decay        - (float)                         decay scale of learning rate when validation performance does not improve or we hit <start_decay_at> epoch
            start_decay_at  - (int)                           decay the learning rate at i-th epoch referenced by <start_decay_at>
            patience        - (int)                           number of epoch after which learning rate will decay if no improvement
            decay_nb        - (int)                           number of learning rate decay
        """
        self.params = list(params)
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.patience = patience
        self.decay_nb = decay_nb
        self.wait = 0
        self.already_decay_nb = 0

        self._makeOptimizer()

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Select the optimizer
    """
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Run the gradient optimization
    """
    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            if param.grad is not None:
                grad_norm += math.pow(param.grad.data.norm(), 2)
            else:
                grad_norm += 0.

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1 and param.grad is not None:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    decay learning rate if validation performance does not improve or we hit the start_decay_at epoch
    """
    def updateLearningRate(self, ppl, epoch):
        """
        Arguments:
            ppl      - (float) the loss value
            epoch    - (int)   the number of trained epoch
        """
        if self.already_decay_nb < self.decay_nb:
            if self.start_decay_at is not None and epoch >= self.start_decay_at:
                self.wait += 1
                if self.wait >= self.patience:
                    self.start_decay = True
                    self.wait = 0
                else:
                    self.start_decay = False
            if self.last_ppl is not None and ppl > self.last_ppl:
                self.wait += 1
                if self.wait >= self.patience:
                    self.start_decay = True
                    self.wait = 0
                else:
                    self.start_decay = False

            if self.start_decay:
                self.already_decay_nb += 1
                self.lr = self.lr * self.lr_decay
                #print("Decaying learning rate to %g" % self.lr)
        else:
            if self.last_ppl is not None and ppl > self.last_ppl:
                self.wait += 1
                if self.wait >= self.patience:
                    raise KeyboardInterrupt
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()