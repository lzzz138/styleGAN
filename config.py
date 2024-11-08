import torch

ROOT                    = "seeprettyface_chs_wanghong/xinggan_face"
START_TRAIN_AT_IMG_SIZE = 8  #The authors start from 8x8 images instead of 4x4
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [2048, 1024, 512, 256, 128, 64, 32, 16]
CHANNELS_IMG            = 3
Z_DIM                   = 256
W_DIM                   = 256
IN_CHANNELS             = 256
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [30] * len(BATCH_SIZES)
