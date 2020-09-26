"""
    Implements all io operations of data.
"""

from utils._libs_ import np, pd, torch, Variable

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Compute the normal std
"""
def normal_std(x):
    """
    Arguments:
        x    - (torch.tensor) dataset for computation
    Returns:
        The normal std tensor
    """
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Get the data generator
"""
def getGenerator(data_name):
    """
    Arguments:
        data_name    - (string) name of data file 
    Returns:
        DataGenerator class that fits data_name
    """
    if 'nasdaq' in data_name:
        return NasdaqGenerator
    else:
        return GeneralGenerator

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
DataGenerator class produces data samples for all models
"""
class DataGenerator():
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Initialization arguments:
        X                  - (numpy.array)  the whole dataset used for training, validation and testing
        mode               - (string)       define the prediction mode of models:
                                                - "immediate":   predict the value at time X_{t+output_T}
                                                - "continuous":  predict the value at time X_{t+output_T-collaborate_span}, ..., X_{t+output_T-1}, X_{t+output_T}, X_{t+output_T+1}, ..., X_{t+output_T+collaborate_span}
        train_share        - (tuple)        two numbers in range (0, 1) showing proportion of training and validation samples 
        input_T            - (int)          number of timesteps in the input
        output_T           - (int)          number of timesteps in the output
        collaborate_span   - (int)          time span for collaborate prediction, only works when mode == "continuous"
        collaborate_stride - (int)          stride for collaborate prediction, only works when mode == "continuous"
        limit              - (int)          maximum number of timesteps-rows in the 'X' DataFrame
        cuda               - (boolean)      whether use gpu to train models
        normalize_pattern  - (int)          define the normalization method:
                                                - 0: use the original data without normalization
                                                - 1: normlized by the maximum absolute value of each column.
                                                - 2: normlized by the mean and std value of each column
    """
    def __init__(self, X, mode, train_share=(.8, .1), input_T=10, output_T=1, collaborate_span=0,
                 collaborate_stride=1, limit=np.inf, cuda=False, normalize_pattern = 2):
        if mode == "continuous" and (output_T <= collaborate_span or collaborate_span <= 0):
            raise Exception("collaborate_span must > 0 and < output_T!")

        self.X = X
        if limit < np.inf: self.X = self.X[:limit]
        self.train_share = train_share
        self.input_T = input_T
        self.output_T = output_T
        self.collaborate_span = collaborate_span
        self.collaborate_stride = collaborate_stride
        self.row_num = self.X.shape[0]
        self.column_num = self.X.shape[1]
        self.n_train = int(self.row_num * train_share[0])
        self.n_valid = int(self.row_num * (train_share[0] + train_share[1]))
        self.n_test = self.row_num
        self.mode = mode
        self.cuda = cuda
        self.normalize_pattern = normalize_pattern

        self.normalize()
        self.split_data()
        self.compute_metrics()

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Normalize data
    """
    def normalize(self):
        if self.normalize_pattern == 0:
            pass
        elif self.normalize_pattern == 1:
            self.maximums = np.max(np.abs(self.X), axis=0)
            self.X = self.X / self.maximums
        elif self.normalize_pattern == 2:
            self.means = np.mean(self.X[:self.n_train], axis=0)
            self.stds = np.std(self.X[:self.n_train], axis=0)
            self.X = (self.X - self.means) / (self.stds + (self.stds == 0) * .001)
        else:
            raise Exception('invalid normalize_pattern')

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Split the training, validation and testing data
    """
    def split_data(self):
        train_range = range(self.input_T + self.output_T - 1, self.n_train)
        valid_range = range(self.n_train, self.n_valid)
        test_range = range(self.n_valid, self.row_num)
        self.train_set = self.batchify(train_range)
        self.valid_set = self.batchify(valid_range)
        self.test_set = self.batchify(test_range)

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Get the X and Y for each set
    """
    def batchify(self, idx_set):
        """
        Arguments:
            idx_set  - (list) index set of data samples
        Returns:
            [X, Y]
        """
        idx_num = len(idx_set)
        if self.mode == "immediate":
            X = torch.zeros((idx_num, self.input_T, self.column_num))
            Y = torch.zeros((idx_num, self.column_num))
            for i in range(idx_num):
                end = idx_set[i] - self.output_T + 1
                start = end - self.input_T
                X[i, :, :] = torch.from_numpy(self.X[start:end, :])
                Y[i, :] = torch.from_numpy(self.X[idx_set[i], :])
        elif self.mode == "continuous":
            X = torch.zeros((idx_num - self.collaborate_span * self.collaborate_stride, self.input_T, self.column_num))
            Y = torch.zeros((idx_num - self.collaborate_span * self.collaborate_stride, self.collaborate_span * 2 + 1, self.column_num))
            for i in range(idx_num - self.collaborate_span * self.collaborate_stride):
                X_end = idx_set[i] - self.output_T + 1
                X_start = X_end - self.input_T
                Y_sample = []
                tmp_span = self.collaborate_span * self.collaborate_stride
                for _ in range(self.collaborate_span * 2 + 1):
                    Y_sample.append(idx_set[i] - tmp_span)
                    tmp_span -= self.collaborate_stride
                X[i, :, :] = torch.from_numpy(self.X[X_start:X_end, :])
                Y[i, :, :] = torch.from_numpy(self.X[Y_sample, :])
        else:
            raise Exception('invalid mode')

        return [X, Y]

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Compute the metrics for evaluating the models
    """
    def compute_metrics(self):
        if self.normalize_pattern == 0:
            if self.mode == "immediate":
                tmp = self.test_set[1]
            else:
                tmp = self.test_set[1][:, self.collaborate_span]
        elif self.normalize_pattern == 1:
            self.maximums = torch.from_numpy(self.maximums).float()
            if self.mode == "immediate":
                tmp = self.test_set[1] * self.maximums.expand(self.test_set[1].size(0), self.column_num)
            else:
                tmp = self.test_set[1][:, self.collaborate_span] * self.maximums.expand(self.test_set[1].size(0), self.column_num)

            if self.cuda:
                self.maximums = self.maximums.cuda()
            self.maximums = Variable(self.maximums)
        else:
            self.means = torch.from_numpy(self.means).float()
            self.stds = torch.from_numpy(self.stds).float()
            tmp_std = self.stds + ((self.stds == 0).float() * 0.001)
            if self.mode == "immediate":
                tmp = self.test_set[1] * tmp_std.expand(self.test_set[1].size(0), self.column_num) + self.means.expand(self.test_set[1].size(0), self.column_num)
            else:
                tmp = self.test_set[1][:, self.collaborate_span] * tmp_std.expand(self.test_set[1].size(0), self.column_num) + self.means.expand(self.test_set[1].size(0), self.column_num)

            if self.cuda:
                self.means = self.means.cuda()
                self.stds = self.stds.cuda()
            self.means = Variable(self.means)
            self.stds = Variable(self.stds)

        self.rse = normal_std(tmp)

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Get the batch data
    """
    def get_batches(self, X, Y, batch_size, shuffle=True):
        """
        Arguments:
            X            - (torch.tensor) input dataset
            Y            - (torch.tensor) ground-truth dataset
            batch_size   - (int)          batch size
            shuffle      - (boolean)      whether shuffle the dataset
        Yields:
            (batch_X, batch_Y)
        """
        length = len(X)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            batch_X = X[excerpt]
            batch_Y = Y[excerpt]
            if (self.cuda):
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()
            yield Variable(batch_X), Variable(batch_Y)
            start_idx += batch_size

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
A GeneralGenerator reads complete data from .txt file without outliers 
"""
class GeneralGenerator(DataGenerator):
     def __init__(self, data_path, mode, train_share=(.8, .1), input_T=10, output_T=1, collaborate_span=0,
                  collaborate_stride=1, limit=np.inf, cuda=False, normalize_pattern = 2):
        X = np.loadtxt(data_path, delimiter=',')
        super(GeneralGenerator, self).__init__(X, mode=mode,
                                               train_share=train_share,
                                               input_T=input_T,
                                               output_T=output_T,
                                               collaborate_span=collaborate_span,
                                               collaborate_stride=collaborate_stride,
                                               limit=limit,
                                               cuda=cuda,
                                               normalize_pattern=normalize_pattern)

# --------------------------------------------------------------------------------------------------------------------------------
"""
A NasdaqGenerator reads samples of NASDAQ 100 stock data.
"""
class NasdaqGenerator(DataGenerator):
    def __init__(self, data_path, mode, train_share=(.8, .1), input_T=10, output_T=1, collaborate_span=0,
                 collaborate_stride=1, limit=np.inf, cuda=False, normalize_pattern = 2):
        X = pd.read_csv(data_path)
        super(NasdaqGenerator, self).__init__(X.values, mode=mode,
                                              train_share=train_share,
                                              input_T=input_T,
                                              output_T=output_T,
                                              collaborate_span=collaborate_span,
                                              collaborate_stride=collaborate_stride,
                                              limit=limit,
                                              cuda=cuda,
                                              normalize_pattern=normalize_pattern)