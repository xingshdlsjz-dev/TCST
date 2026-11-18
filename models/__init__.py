from .cnn1d import CNN1D
from .deepconvlstm import DeepConvLSTM
from .itransformer import ITransformerHAR
from .lstm import LSTMClassifier
from .patchtst import PatchTSTHAR
from .resnet1d import ResNet1D
from .inceptiontime import InceptionTime
from .tcn import TCN
from .grufcn import GRUFCN
from .timesnet import TimesNetHAR


MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "deepconvlstm": DeepConvLSTM,
    "lstm": LSTMClassifier,
    "patchtst": PatchTSTHAR,
    "itransformer": ITransformerHAR,
    "resnet1d": ResNet1D,
    "inceptiontime": InceptionTime,
    "tcn": TCN,
    "grufcn": GRUFCN,
    "timesnet": TimesNetHAR,
}
