#-*-coding:utf-8-*- 
import os
import sys
import argparse
import logging
import random
import json
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from utils.parallel import DistributedDataParallelModel as DistributedDataParallel
from torch.nn.parallel.distributed import DistributedDataParallel as DistributedDataParallel
from torch.nn.parallel import DataParallel
from model.Seq2Seq_entire_inter import Seq2SeqClassifier
from model.DecoderRNN_entire_inter import DecoderRNN
# from utils.dataset_entire import LipReadingDataSet, collect_fn, q_collect_fn  ####读取灰度图像
from utils.dataset_RGB_entire_inter import  LipReadingDataSet, collect_fn, q_collect_fn, project_collect_fn   #####读取 RGB 图像
from utils.checkpoint import Checkpoint
from trainer_entire_inter import SupervisedTrainer
from optimizer import Optimizer
from loss import Perplexity
from decoder import GreedyDecoder
from model.jointLoss_entire_inter import JointLoss
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils import transforms

###### 多GPU设置
def setup(rank, world_size):
    # initialize the process group
    # dist.init_process_group('nccl', init_method='tcp://10.214.29.109:12345', rank=rank, world_size=world_size)
    dist.init_process_group('nccl', init_method='tcp://localhost:23457', rank=rank, world_size=world_size)
    # dist.init_process_group(backend='nccl', init_method='env://')


def cleanup():
    dist.destroy_process_group()


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


torch.multiprocessing.set_sharing_strategy('file_system')

# Train, validation, vocab file，文件
train_index_filename = "DatasetFile/small_dataset_resize_split_translation/train_modified.txt"   ######small_dataset_resize_split1  or  split5_dialogue
val_index_filename = "DatasetFile/small_dataset_resize_split_translation/test_modified.txt"
ch_vocab_filename = "DatasetFile/small_dataset_resize_split_translation/a_vocab.txt"
q_ch_vocab_filename = 'DatasetFile/small_dataset_resize_split_translation/q_vocab.txt'
history_json_train = "DatasetFile/small_dataset_resize_split_translation/history_train.json"
history_json_test = "DatasetFile/small_dataset_resize_split_translation/history_test.json"

# Argument parameter
parser = argparse.ArgumentParser(description='CSL Training!')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001) #0.001,0.00001
parser.add_argument('--weight_decay', type=float, default=1e-03) #
parser.add_argument('--rank', type=int, default=-1)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--bs', type=int, default=int(2))
parser.add_argument('--frame_rate', type=int, default=25)
parser.add_argument('--expt_dir', action='store', dest='expt_dir',
                    default='/ssd2/test')
parser.add_argument('--pretrained', action='store', dest='pre_trained',default=None)
# parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='/ssd2/gaoliqing/NCAA_response/Rprop/model_checkpoint/save_15.pth')   ####load用这行
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',default=None)
parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')
parser.add_argument("--multi_gpu", type=bool, default=False, help='Whether to use multi-gpu')
parser.add_argument("--method", default=None, help='whether to use beam-search')
opt = parser.parse_args()


if opt.multi_gpu:
    print(opt.local_rank)
    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    opt.rank = dist.get_rank()
    opt.world_size = dist.get_world_size()
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    opt.local_rank = 0

# Set CUDA device
seed = 2018216019
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#####定义log文件
if not os.path.exists(opt.expt_dir):
    os.mkdir(opt.expt_dir)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(opt.expt_dir, str(opt.local_rank) + '_log.txt'), mode='a')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

if opt.local_rank == 0:
    logger.info(opt)


train_transforms = transforms.Compose([
    # transforms.Resize((112, 112)),
    # transforms.RandomCrop(110),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


val_transforms = transforms.Compose([
    # transforms.Resize((112, 112)),
    # transforms.RandomCrop(140),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ch_vocab = ['<unk>', '<pad>', '<eos>', '<sos>']
with open(ch_vocab_filename, encoding="GBK") as f:    #### GBK or utf-8
    for line in f.readlines():
        ch_vocab.append(line.strip())

tgt_eos_id = 2
tgt_sos_id = 3

# question vocab list
q_ch_vocab = ['<unk>', '<pad>']
with open(q_ch_vocab_filename, encoding="GBK") as f:   #### GBK or utf-8
    for line in f.readlines():
        q_ch_vocab.append(line.strip())

# CTC Vocab list
ctc_ch_vocab = ['-']
with open(ch_vocab_filename, encoding="GBK") as f:   #### GBK or utf-8
    for line in f.readlines():
        ctc_ch_vocab.append(line.strip())




# history_json
history_train = json.load(open(history_json_train, 'r', encoding="GBK"))
history_test = json.load(open(history_json_test, 'r', encoding="GBK"))
# history = json.load(open(history_json, 'r', encoding="GBK"))


# dialogue_vocab list
dialogue_ch_vocab = []
#### CSL
for k, v in enumerate(history_train):
    for k_, v_ in enumerate(history_train[v]):
        if v_ not in dialogue_ch_vocab:
            dialogue_ch_vocab.append(v_)




#### loss 定义
criterion = JointLoss(ce_weight=1, ctc_weight=0, kl_weight=0.4, ctc_blank_label=0, cross_entroy_loss_label=1)



# Prepare the decoder
ch_greedy_decoder = GreedyDecoder(labels=ch_vocab)

# Seq2Seq Module
encoder_rnn_input_size_ = 512
encoder_hidden_size_ = 256
encoder_bidirectional = True
decoder_rnn_input_size_ = 256
decoder_hidden_size_ = 256
theme_size = 20    #   20 or 5
dt_memory_size = 100
dtqv_memory_size = 100

ch_decoder_rnn = DecoderRNN(len(ch_vocab),
                            encoder_hidden_size_ * 2 if encoder_bidirectional else encoder_hidden_size_,
                            n_layers=1,
                            dropout_p=0,
                            use_attention=True,
                            bidirectional=encoder_bidirectional,
                            eos_id=tgt_eos_id,
                            sos_id=tgt_sos_id,
                            use_Question=True)

classifier = Seq2SeqClassifier(encoder_rnn_size=encoder_rnn_input_size_,
                               encoder_hidden_size=encoder_hidden_size_,
                               ch_decoder_rnn=ch_decoder_rnn,
                               vocab_size=len(ch_vocab),
                               question_vocab_size=len(q_ch_vocab),
                               use_question_model=True,
                               criterion=criterion,
                               batch_size=opt.bs,
                               decoder_rnn_input_size=decoder_rnn_input_size_,
                               theme_size=theme_size,
                               dt_memory_size=dt_memory_size,
                               dtqv_memory_size=dtqv_memory_size,
                               )


if torch.cuda.is_available():
    classifier.cuda()
    criterion.cuda()

if opt.multi_gpu:
    classifier = DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)


if opt.local_rank == 0:
    logger.info(classifier)

if opt.pre_trained is not None:
    print('load %s ' % (opt.pre_trained))

    pre_trained_dict = torch.load(opt.pre_trained)
    classifier_dict = classifier.state_dict()

    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if
                        k in classifier_dict and v.size() == classifier_dict[k].size()}
    for k, v in pre_trained_dict.items():
        print(k)
    print('----------------------------')
    classifier_dict.update(pre_trained_dict)
    classifier.load_state_dict(classifier_dict)

# Optimizer
base_optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)

# Load the checkpoint 从某个epoch开始训练
if opt.load_checkpoint is not None or opt.resume:
    if opt.load_checkpoint is not None:
        checkpoint_path = opt.load_checkpoint
    else:
        checkpoint_path = Checkpoint.get_latest_checkpoint(opt.expt_dir)
    if opt.local_rank == 0:
        logger.info("loading checkpoint from {}".format(checkpoint_path))
    #### 之前的
    # resume_checkpoint = Checkpoint.load(checkpoint_path)
    # classifier.load_state_dict(resume_checkpoint.model_state, strict=False)
    # base_optimizer.load_state_dict(resume_checkpoint.optimizer_state)
    # start_epoch = resume_checkpoint.epoch
    # min_er = resume_checkpoint.min_er
    # min_cost = resume_checkpoint.min_cost
    #### 之前的
    checkpoint = torch.load(checkpoint_path)
    # classifier.load_state_dict(checkpoint['state_dict'], strict=False)
    classifier.load_state_dict(checkpoint['state_dict'])
    base_optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    min_er = checkpoint['min_er']
    min_cost = checkpoint['min_cost']


else:
    start_epoch = 0
    min_er = 9999999999
    min_cost = 9999999999

batch_size = opt.bs
# Set the Train and Validation data_loader
train_data_set = LipReadingDataSet(index_file=train_index_filename, ch_vocab=ch_vocab, q_ch_vocab=q_ch_vocab,
                                   history=history_train, history_window=2,
                                   transforms=train_transforms,
                                   dialogue_ch_vocab=dialogue_ch_vocab,mode="train")

if opt.multi_gpu:
    train_sampler = DistributedSampler(train_data_set)
else:
    train_sampler = None


train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True,
                          collate_fn=project_collect_fn, num_workers=0, sampler=train_sampler, drop_last=True, pin_memory=True)

val_data_set = LipReadingDataSet(index_file=val_index_filename, ch_vocab=ch_vocab, q_ch_vocab=q_ch_vocab,
                                    history=history_test, history_window=2,
                                 transforms=val_transforms,
                                 dialogue_ch_vocab=dialogue_ch_vocab,mode="test")

val_sampler = None
val_loader = DataLoader(val_data_set, batch_size=batch_size, collate_fn=project_collect_fn, num_workers=0,
                        sampler=val_sampler, drop_last=True, shuffle=False, pin_memory=True)

# Set the optimizer
train_num = len(train_data_set)
evaluate_step = train_num // (batch_size * opt.world_size )
print_step = evaluate_step // 10

####学习率更新
scheduler_plateau = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=10, gamma=0.1)

optimizer = Optimizer(base_optimizer, max_grad_norm=5)
optimizer.set_scheduler(scheduler_plateau)


# Train the seq2seq model
t = SupervisedTrainer(criterion=criterion,
                      ch_greedy_decoder=ch_greedy_decoder, optimizer=optimizer, print_every=print_step,
                      evaluate_every=evaluate_step,
                      expt_dir=opt.expt_dir,
                      writer_path=opt.expt_dir,
                      train_sampler=train_sampler, rank=opt.local_rank, world_size=opt.world_size,
                      logger=logger)

seq2seq_model = t.train(classifier, train_loader, dev_data_loader=val_loader, n_epochs=600,
                        start_epoch=start_epoch, min_er=min_er, min_cost=min_cost, method=opt.method,batch_size=opt.bs)

