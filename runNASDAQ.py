from utils.data_io import getGenerator
from utils.args import Args, list_of_param_dicts
from models.models import MLCNN
from models.model_runner import ModelRunner
import gc

param_dict = dict(
    data = ['data/nasdaq100_padding.csv'],
    mode = ['continuous'],
    collaborate_span = [2],
    collaborate_stride = [1],
    train_share = [(0.9, 0.05)],
    input_T = [60,180],
    n_CNN = [5],
    kernel_size = [3,5],
    hidCNN = [5,10],
    hidRNN = [50,100,200],
    dropout = [0.3,0.5],
    highway_window = [8],
    clip = [10.],
    epochs = [100],
    batch_size = [128],
    seed = [54321],
    gpu = [0],
    cuda = [True],
    optim = ['adam'],
    lr = [0.001],
    output_T = [3, 6, 12],
    L1Loss = [False],
    normalize = [2]
)

if __name__ == '__main__':
    params = list_of_param_dicts(param_dict)
    for param in params:
        cur_args = Args(param)
        generator = getGenerator(cur_args.data)
        data_gen = generator(cur_args.data, cur_args.mode,
                             train_share=cur_args.train_share, input_T=cur_args.input_T, output_T=cur_args.output_T,
                             collaborate_span=cur_args.collaborate_span, collaborate_stride=cur_args.collaborate_stride,
                             cuda=cur_args.cuda, normalize_pattern=cur_args.normalize)
        runner = ModelRunner(cur_args, data_gen, None)
        runner.model = MLCNN(cur_args, data_gen)
        runner.run()
        runner.getMetrics()
        del runner
        gc.collect()