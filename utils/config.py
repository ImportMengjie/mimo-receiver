import os
import torch

USE_GPU = True

USE_GPU = USE_GPU and torch.cuda.is_available()

# save files
DATA = 'data'
RESULT = 'result'

DENOISING_RESULT = os.path.join(RESULT, 'denoising')
DETECTION_RESULT = os.path.join(RESULT, 'detection')
INTERPOLATION_RESULT = os.path.join(RESULT, 'interpolation')

DENOISING_RESULT_IMG = os.path.join(DENOISING_RESULT, 'img')
DETECTION_RESULT_IMG = os.path.join(DETECTION_RESULT, 'img')
INTERPOLATION_RESULT_IMG = os.path.join(INTERPOLATION_RESULT, 'img')

ANALYSIS_BATCH_SIZE = 100
