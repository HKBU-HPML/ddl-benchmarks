from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
#from torchvision import models
from transformers import BertTokenizer, BertForPreTraining, BertConfig
from transformers import AdamW
#import byteps.torch as bps
import byteps_distributed_optimizer as bps
import timeit
import numpy as np
import os, sys

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='bert',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--sentence-len', type=int, default=64,
                    help='input sentence len')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bps.init()

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())

cudnn.benchmark = True

if args.model == 'bert_base':
    config = BertConfig.from_json_file('bert_base_config.json')
else:
    config = BertConfig.from_json_file('bert_config.json')
#config = BertConfig.from_json_file('bert_config.json')
# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)

vocab_size=config.vocab_size
#tokenizer = BertTokenizer.from_pretrained(pretrained_path)
#model = BertForPreTraining.from_pretrained(pretrained_path)
model = BertForPreTraining(config)


# Set up standard model.
#model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

#optimizer = AdamW(model.parameters(),
#        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#        )

# BytePS: (optional) compression algorithm.
compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fake data
max_len = args.sentence_len
batch_size = args.batch_size
#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
input_ids = (torch.rand(batch_size, max_len) * 2000).long()
attention_masks = torch.rand(batch_size, max_len).long()
token_type_ids = torch.rand(batch_size, max_len).long()
position_ids = (torch.rand(batch_size, max_len) * 10).long()
next_sentence_label = torch.rand(batch_size, 1).long()
masked_lm_labels = torch.rand(batch_size, max_len).long()
batch = (input_ids, attention_masks, token_type_ids, position_ids, next_sentence_label, masked_lm_labels)
if args.cuda:
    batch = tuple(item.cuda() for item in batch)
input_ids, attention_masks, token_type_ids, position_ids, next_sentence_label, masked_lm_labels = batch

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss
criterion = BertPretrainingCriterion(vocab_size)


def benchmark_step():
    optimizer.zero_grad()
    prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_label)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if bps.local_rank() != 0:
        return
    print(s, end='\n' if nl else '')
    sys.stdout.flush()


log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, bps.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('BERT Large Pretraining, Sentence len: %d' % max_len)
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    sen_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f sentences/sec per GPU' % (x, sen_sec))
    img_secs.append(sen_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Sen/sec per %s: %.1f +-%.1f' % ('GPU', img_sec_mean, img_sec_conf))
log('Total sen/sec on %d %s(s): %.1f +-%.1f' %
    (bps.size(), 'GPU', bps.size() * img_sec_mean, bps.size() * img_sec_conf))

