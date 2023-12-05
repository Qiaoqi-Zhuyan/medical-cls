import os
import logging
import random
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

import torch
from torch import nn
from timm.utils import accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from config.model_config import get_config
from datas.build import build_loader
from tools.build_optimizer import build_optimizer

from model.ConvNext_official_impl import convnext_tiny, convnext_small
from model.convnext import ConvUBCModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
config = get_config()


# logger


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.logger_name)
    ]
)
logging.basicConfig(stream=None)
logger = logging.getLogger("training_logger")

# model config
#model = convnext_tiny(pretrained=False, num_classes=5, drop_path_rate=0.1)
#model = convnext_small(pretrained=False, num_classes=5, drop_path_rate=0.5)
model = ConvUBCModel('convnext_base_in22k', 5)
model = model.to("cuda")
optimizer = build_optimizer(model, config)

if config.use_labelsmooth_loss:
    loss_fn = SoftTargetCrossEntropy()
else:
    loss_fn = nn.CrossEntropyLoss()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(config):

    print(f"device {device}")
    print(f"training launched ...")

    print("processing datasets")
    train_data_loader, val_data_loader, mixup_fn = build_loader(config)

    logger.info(
        f'model: {config.model_name}\n'
        f'logger_name: {config.logger_name}\n'
        f'weights: {config.save_model_name}'
        f'config:\n'
        f'\tbatch_size: {config.batch_size}\n'
        f'\tepochs: {config.train_epoches}\n'
        f'\tlr: {config.lr}\n'
        f'\toptimizer: {config.optimizer}\n'
        f'\tuse_labelsmooth_loss: {config.use_labelsmooth_loss}'
    )

    epoch_num = config.train_epoches
    for epoch in range(epoch_num):

        train_correct = 0
        train_total = 0

        model.train()
        with tqdm(train_data_loader, desc="Train") as t:
            for x, y in t:
                t.set_description(f"Epoch [{epoch} / {epoch_num}]")
                x = x.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)

                if mixup_fn is not None:
                    x, y = mixup_fn(x, y)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                #train_timm_acc = '{:.4f}'.format(accuracy(y_pred, y, topk=(1,)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    #y_ = torch.argmax(y_pred, dim=1)

                    train_loss = loss.item()

                    #train_correct += (y_ == y).sum().item()
                    #train_total += y.size(0)
                    #train_acc = '{:/4f}'.format(train_correct / train_total)

                    #y = y.to("cpu")
                    #y_ = y_.to("cpu")

                    #train_balanced_acc = '{:.4f}'.format(balanced_accuracy_score(y, y_))

                t.set_postfix(train_loss=train_loss)
                logger.info(f'Train:'
                            f'Epoch: [{epoch} / {epoch_num}]\t'
                            f'Training Loss: {train_loss}\t'
                            )

                torch.save(model.state_dict(), config.save_model_name)

        val_correct = 0
        val_total = 0

        model.eval()
        with torch.no_grad():
            with tqdm(val_data_loader, desc="Val") as t2:
                for x, y in t2:
                    x = x.to('cuda', non_blocking=True)
                    y = y.to('cuda', non_blocking=True)
                    y_p = model(x)

                    pred = torch.argmax(y_p, dim=1)

                    val_total += y.size(0)
                    val_correct += (pred == y).sum().item()
                    val_acc = '{:.4f}'.format(val_correct / val_total)
                    top_1 = accuracy(y_p, y, topk=(1, ))

                    top_1_acc = top_1.item() if isinstance(top_1, torch.Tensor) else top_1[0]

                    val_timm_acc = '{:.4f}'.format(top_1_acc)


                    y = y.to('cpu')
                    pred = pred.to('cpu')
                    balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

                    t2.set_postfix(val_acc=val_acc,val_timm_acc=val_timm_acc ,val_BA=balanced_accuracy)

                    logger.info(f'Val:'
                                f'Epoch: [{epoch} / {epoch_num}, \t'
                                f'Val Accuracy: {val_acc},\t'
                                f'Val Balanced Accuracy: {balanced_accuracy}\t'
                                f'Val Timm Acc: {val_timm_acc}'
                                )


if __name__ == "__main__":
    set_seed(42)
    train(config)


