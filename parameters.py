import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'
bpeCodesFileName = 'bpeCodes'

device = torch.device("cuda:0")
# device = torch.device("cpu")

learning_rate = 0.001
batchSize = 64
clip_grad = 2.0
dropout = 0.1
nhead = 8
decoder_layers = 6

# embed_size = 128 # Small Model
# dim_feedforward = 512 # Small Model

# embed_size = 256 # Mid Model
# dim_feedforward = 1024 # Mid Model

embed_size = 512 # Large Model
dim_feedforward = 1536 # Large Model

maxEpochs = 10
log_every = 10
test_every = 2000
warmup_steps = 4000