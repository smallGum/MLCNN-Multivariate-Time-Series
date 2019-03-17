"""
    Define all models.
"""

from utils._libs_ import torch, nn, F, np

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Multi-Level Construal Neural Network
"""
class MLCNN(nn.Module):
    def __init__(self, args, data):
        """
        Initialization arguments:
            args   - (object)  parameters of model
            data   - (DataGenerator object) the data generator
        """
        super(MLCNN, self).__init__()
        self.use_cuda = args.cuda
        self.input_T = args.input_T
        self.idim = data.column_num
        self.kernel_size = args.kernel_size
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.hw = args.highway_window
        self.collaborate_span = args.collaborate_span
        self.cnn_split_num = int(args.n_CNN / (self.collaborate_span * 2 + 1))
        self.n_CNN = self.cnn_split_num * (self.collaborate_span * 2 + 1)

        self.dropout = nn.Dropout(p = args.dropout)
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for i in range(self.n_CNN):
            if i == 0:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.idim), padding=(self.kernel_size//2, 0))
            else:
                tmpconv = nn.Conv2d(1, self.hidC, kernel_size=(self.kernel_size, self.hidC), padding=(self.kernel_size//2, 0))
            self.convs.append(tmpconv)
            self.bns.append(nn.BatchNorm2d(self.hidC))
        self.shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.target_lstm = nn.LSTM(self.hidC, self.hidR)
        self.linears = nn.ModuleList([])
        self.highways = nn.ModuleList([])
        for i in range(self.collaborate_span * 2 + 1):
            self.linears.append(nn.Linear(self.hidR, self.idim))
            if (self.hw > 0):
                self.highways.append(nn.Linear(self.hw * (i+1), 1))

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Forward propagation
    """
    def forward(self, x):
        """
        Arguments:
            x   - (torch.tensor) input data
        Returns:
            res - (torch.tensor) result of prediction
        """
        regressors = []
        currentR = torch.unsqueeze(x, 1)
        for i in range(self.n_CNN):
            currentR = self.convs[i](currentR)
            currentR = self.bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            currentR = torch.squeeze(currentR, 3)
            if (i + 1) % self.cnn_split_num == 0:
                regressors.append(currentR)
                currentR = self.dropout(currentR)
            if i < self.n_CNN - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)

        shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None
        self.shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            cur_R = regressors[i].permute(2,0,1).contiguous()
            _, (cur_result, cur_state) = self.shared_lstm(cur_R)
            if i == self.collaborate_span:
                target_R = cur_R
                target_h = cur_result
                target_c = cur_state
            cur_result = self.dropout(torch.squeeze(cur_result, 0))
            shared_lstm_results.append(cur_result)

        self.target_lstm.flatten_parameters()
        _, (target_result, _) = self.target_lstm(target_R, (target_h, target_c))
        target_result = self.dropout(torch.squeeze(target_result, 0))

        res = None
        for i in range(self.collaborate_span * 2 + 1):
            if i == self.collaborate_span:
                cur_res = self.linears[i](target_result)
            else:
                cur_res = self.linears[i](shared_lstm_results[i])
            cur_res = torch.unsqueeze(cur_res, 1)
            if res is not None:
                res = torch.cat((res, cur_res), 1)
            else:
                res = cur_res

        #highway
        if (self.hw > 0):
            highway = None
            for i in range(self.collaborate_span * 2 + 1):
                z = x[:, -(self.hw * (i+1)):, :]
                z = z.permute(0,2,1).contiguous().view(-1, self.hw * (i+1))
                z = self.highways[i](z)
                z = z.view(-1, self.idim)
                z = torch.unsqueeze(z, 1)
                if highway is not None:
                    highway = torch.cat((highway, z), 1)
                else:
                    highway = z
            res = res + highway

        return res