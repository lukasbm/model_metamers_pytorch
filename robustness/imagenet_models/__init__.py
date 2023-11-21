from .alexnet import *
from .alexnet_reduced_aliasing import *
from .cornet_s import *
from .hmax import *
# For the ipcl models
from .ipcl_alexnet_gn import ipcl_alexnet_gn
from .resnet import *
from .resnet_openselfsup_transfer import *
# Shape trained models
from .texture_shape_models import *
from .timm_resnet_gelu import resnet50_gelu, swsl_resnext101_32x8d
from .vgg import *
# For VITs, CLIP, and SWSL
from .vision_transformer import *
from .vonenet import *
