"""
    Define ModelRunner class to run the model.
"""

from utils._libs_ import math, time, torch, nn, np
from utils.data_io import DataGenerator
from models.optimize import Optimize

class ModelRunner():
    def __init__(self, args, data_gen, model):
        """
        Initialization arguments:
            args       - (object)                 parameters of model
            data_gen   - (DataGenerator object)   the data generator
            model      - (torch.nn.Module object) the model to be run
        """
        self.args = args
        self.data_gen = data_gen
        self.model = model
        self.best_rmse = None
        self.best_rse = None
        self.best_mae = None
        self.running_times = []
        self.train_losses = []

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Train the model
    """
    def train(self):
        self.model.train()
        total_loss = 0
        n_samples = 0

        for X, Y in self.data_gen.get_batches(self.data_gen.train_set[0], self.data_gen.train_set[1], self.args.batch_size, True):
            self.model.zero_grad()
            output = self.model(X)

            if self.data_gen.normalize_pattern == 0:
                loss = self.criterion(output, Y)
            elif self.data_gen.normalize_pattern == 1:
                if self.data_gen.mode == "immediate":
                    maximums = self.data_gen.maximums.expand(output.size(0), self.data_gen.column_num)
                    loss = self.criterion(output * maximums, Y * maximums)
                else:
                    tmp_maximums = torch.unsqueeze(self.data_gen.maximums, 0)
                    tmp_maximums = torch.unsqueeze(tmp_maximums, 1)
                    new_maximums = tmp_maximums.expand(output.size(0), output.size(1), self.data_gen.column_num)
                    loss = self.criterion(output * new_maximums, Y * new_maximums)
            else:
                if self.data_gen.mode == "immediate":
                    means = self.data_gen.means.expand(output.size(0), self.data_gen.column_num)
                    tmp_stds = self.data_gen.stds + ((self.data_gen.stds == 0).float() * 0.001)
                    stds = tmp_stds.expand(output.size(0), self.data_gen.column_num)
                    loss = self.criterion(output * stds + means, Y * stds + means)
                else:
                    tmp_stds = self.data_gen.stds + ((self.data_gen.stds == 0).float() * 0.001)
                    tmp_means = torch.unsqueeze(self.data_gen.means, 0)
                    tmp_means = torch.unsqueeze(tmp_means, 1)
                    tmp_stds = torch.unsqueeze(tmp_stds, 0)
                    tmp_stds = torch.unsqueeze(tmp_stds, 1)
                    new_means = tmp_means.expand(output.size(0), output.size(1), self.data_gen.column_num)
                    new_stds = tmp_stds.expand(output.size(0), output.size(1), self.data_gen.column_num)
                    loss = self.criterion(output * new_stds + new_means, Y * new_stds + new_means)

            loss.backward()
            grad_norm = self.optim.step()
            total_loss += loss.item()
            if self.data_gen.mode == "immediate":
                n_samples += (output.size(0) * self.data_gen.column_num)
            else:
                n_samples += (output.size(0) * output.size(1) * self.data_gen.column_num)

        return total_loss / n_samples

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Valid the model while training
    """
    def evaluate(self, mode='valid'):
        """
        Arguments:
            mode   - (string) 'valid' or 'test'
        """
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        if mode == 'valid':
            tmp_X = self.data_gen.valid_set[0]
            tmp_Y = self.data_gen.valid_set[1]
        elif mode == 'test':
            tmp_X = self.data_gen.test_set[0]
            tmp_Y = self.data_gen.test_set[1]
        else:
            raise Exception('invalid evaluation mode')

        for X, Y in self.data_gen.get_batches(tmp_X, tmp_Y, self.args.batch_size, False):
            output = self.model(X)

            if self.data_gen.normalize_pattern == 0:
                L1_loss = self.evaluateL1(output, Y).item()
                L2_loss = self.evaluateL2(output, Y).item()
                if predict is None:
                    predict = output
                    test = Y
                else:
                    predict = torch.cat((predict, output))
                    test = torch.cat((test, Y))
            elif self.data_gen.normalize_pattern == 1:
                maximums = self.data_gen.maximums.expand(output.size(0), self.data_gen.column_num)
                if self.data_gen.mode == "immediate":
                    L1_loss = self.evaluateL1(output * maximums, Y * maximums).item()
                    L2_loss = self.evaluateL2(output * maximums, Y * maximums).item()
                    if predict is None:
                        predict = output * maximums
                        test = Y * maximums
                    else:
                        predict = torch.cat((predict, output * maximums))
                        test = torch.cat((test, Y * maximums))
                else:
                    L1_loss = self.evaluateL1(output[:, self.data_gen.collaborate_span] * maximums, Y[:, self.data_gen.collaborate_span] * maximums).item()
                    L2_loss = self.evaluateL2(output[:, self.data_gen.collaborate_span] * maximums, Y[:, self.data_gen.collaborate_span] * maximums).item()
                    if predict is None:
                        predict = output[:, self.data_gen.collaborate_span] * maximums
                        test = Y[:, self.data_gen.collaborate_span] * maximums
                    else:
                        predict = torch.cat((predict, output[:, self.data_gen.collaborate_span] * maximums))
                        test = torch.cat((test, Y[:, self.data_gen.collaborate_span] * maximums))

            else:
                means = self.data_gen.means.expand(output.size(0), self.data_gen.column_num)
                tmp_stds = self.data_gen.stds + ((self.data_gen.stds == 0).float() * 0.001)
                stds = tmp_stds.expand(output.size(0), self.data_gen.column_num)
                if self.data_gen.mode == "immediate":
                    L1_loss = self.evaluateL1(output * stds + means, Y * stds + means).item()
                    L2_loss = self.evaluateL2(output * stds + means, Y * stds + means).item()
                    if predict is None:
                        predict = output * stds + means
                        test = Y * stds + means
                    else:
                        predict = torch.cat((predict, output * stds + means))
                        test = torch.cat((test, Y * stds + means))
                else:
                    L1_loss = self.evaluateL1(output[:, self.data_gen.collaborate_span] * stds + means, Y[:, self.data_gen.collaborate_span] * stds + means).item()
                    L2_loss = self.evaluateL2(output[:, self.data_gen.collaborate_span] * stds + means, Y[:, self.data_gen.collaborate_span] * stds + means).item()
                    if predict is None:
                        predict = output[:, self.data_gen.collaborate_span] * stds + means
                        test = Y[:, self.data_gen.collaborate_span] * stds + means
                    else:
                        predict = torch.cat((predict, output[:, self.data_gen.collaborate_span] * stds + means))
                        test = torch.cat((test, Y[:, self.data_gen.collaborate_span] * stds + means))

            total_loss_l1 += L1_loss
            total_loss += L2_loss
            n_samples += (output.size(0) * self.data_gen.column_num)

        mse = total_loss / n_samples
        rse = math.sqrt(total_loss / n_samples) / self.data_gen.rse
        mae = total_loss_l1 / n_samples

        #if mode == 'test':
        #    self.predict = predict.data.cpu().numpy()
        #    self.ground_truth = test.data.cpu().numpy()

        return mse, rse, mae

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Run the model
    """
    def run(self):
        #print(self.model)
        use_cuda = self.args.gpu is not None
        if use_cuda:
            if type(self.args.gpu) == list:
                self.model = nn.DataParallel(self.model, device_ids=self.args.gpu)
            else:
                torch.cuda.set_device(self.args.gpu)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(self.args.seed)
        if use_cuda: self.model.cuda()

        self.nParams = sum([p.nelement() for p in self.model.parameters()])

        if self.args.L1Loss:
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            self.criterion = nn.MSELoss(reduction='sum')
        self.evaluateL1 = nn.L1Loss(reduction='sum')
        self.evaluateL2 = nn.MSELoss(reduction='sum')
        if use_cuda:
            self.criterion = self.criterion.cuda()
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()

        self.optim = Optimize(self.model.parameters(), self.args.optim, self.args.lr, self.args.clip)

        best_valid_mse = float("inf")
        best_valid_rse = float("inf")
        best_valid_mae = float("inf")
        best_test_mse = float("inf")
        best_test_rse = float("inf")
        best_test_mae = float("inf")

        tmp_losses = []
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                train_loss = self.train()
                self.running_times.append(time.time() - epoch_start_time)
                tmp_losses.append(train_loss)
                val_mse, val_rse, val_mae = self.evaluate()
                if val_mse < best_valid_mse:
                    best_valid_mse = val_mse
                    best_valid_rse = val_rse
                    best_valid_mae = val_mae

                self.optim.updateLearningRate(val_mse, epoch)

                test_mse, test_rse, test_mae = self.evaluate(mode='test')
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    best_test_rse = test_rse
                    best_test_mae = test_mae
        except KeyboardInterrupt:
            pass

        self.train_losses.append(tmp_losses)
        # In our experiment, the validation set actually serves as part of the test set.
        # Therefore, we average the best valid and test result for model comparison.
        final_best_mse = (best_valid_mse + best_test_mse) / 2.0
        final_best_rse = (best_valid_rse + best_test_rse) / 2.0
        final_best_mae = (best_valid_mae + best_test_mae) / 2.0
        # otherwise, uncomment the following codes to see the best test result only
        #final_best_mse = best_test_mse
        #final_best_rse = best_test_rse
        #final_best_mae = best_test_mae

        self.best_rmse = np.sqrt(final_best_mse)
        self.best_rse = final_best_rse
        self.best_mae = final_best_mae

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Compute and output the metrics
    """
    def getMetrics(self):
        print('-' * 100)
        print()

        print('* number of parameters: %d' % self.nParams)
        for k in self.args.__dict__.keys():
            print(k, ': ', self.args.__dict__[k])

        running_times = np.array(self.running_times)
        #train_losses = np.array(self.train_losses)

        print("time: sum {:8.7f} | mean {:8.7f}".format(np.sum(running_times), np.mean(running_times)))
        #print("loss trend: ", self.train_losses)
        print("rmse: {:8.7f}".format(self.best_rmse))
        print("rse: {:8.7f}".format(self.best_rse))
        print("mae: {:8.7f}".format(self.best_mae))
        print()