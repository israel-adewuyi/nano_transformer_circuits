import os
import sys
import time
import torch


from functools import partial
from yacs.config import CfgNode as CN
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from attention_only_transformer.train.trainer import TransformerTrainer
from attention_only_transformer.train.utils import set_seed, setup_logging
from attention_only_transformer.one_layer_attn_transformer import OneLayerTransformer


def get_config():
    
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3457
    C.system.work_dir = 'out/artefacts'

    # model
    C.model = OneLayerTransformer.get_default_config()
    C.model.vocab_size = 50257 + 1  # The +1 is for the extra start token
    C.model.n_embd = 768
    C.model.n_head = 12
    C.model.attn_pdrop = 0.1
    C.model.block_size = 2048

    # trainer
    C.trainer = TransformerTrainer.get_default_config()
    C.trainer.block_size = 2048
    C.trainer.batch_size = 32
    C.trainer.micro_batch_size = 8

    C.trainer.learning_rate = 2e-4
    C.trainer.decay_lr = True
    C.trainer.warmup_iters = 1000
    C.trainer.lr_decay_iters = 40000
    C.trainer.min_lr = 1e-5
    C.trainer.max_iters = 50000

    C.trainer.start_token = 50257

    C.model.device = C.trainer.device
    return C


def batch_end_callback(trainer, writer, config):
    if trainer.iter_num % 10 == 0:
        # log training loss
        writer.add_scalar('train_loss', trainer.loss, trainer.iter_num)
        writer.add_scalar('learning_rate', trainer.current_lr, trainer.iter_num)

    if trainer.iter_num % 500 == 0:
        # log validation loss
        val_loss = trainer.validate()
        writer.add_scalar('val_loss', val_loss, trainer.iter_num)
        print(f"iter {trainer.iter_num} val loss: {val_loss}")
    
    if trainer.iter_num % 2000 == 0:
        print('saving latest model at iter', trainer.iter_num)
        # save the latest model
        ckpt_path = os.path.join(config.system.work_dir, f"latest_model_{trainer.iter_num}.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)


def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    config = get_config()
    print(config)

    set_seed(config.system.seed)
    setup_logging(config)

    # new writer for each run based on time
    writer = SummaryWriter(os.path.join(config.system.work_dir, 'tensorboard', time.strftime("%Y-%m-%d_%H-%M-%S")))

    data_dir = "../data/openwebtext"
    if not os.path.exists(data_dir):
        raise ValueError("data not found, please run openwebtext.py")

    # construct the model
    config.model.block_size = config.trainer.block_size
    model = OneLayerTransformer(config.model)

    # construct the trainer object
    trainer = TransformerTrainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback('on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    print(model)

    trainer.run()

if __name__ == '__main__':
    train()