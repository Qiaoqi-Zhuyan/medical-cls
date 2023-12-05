import logging

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from timm.utils import accuracy

from trainer import config, model
from datas.build import UBCDataset_Origin, build_transform

logger_name = f"test--{config.logger_name}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logger_name)
    ]
)
logging.basicConfig(stream=None)
logger = logging.getLogger("test_logger")


def build_test_dataloader(config):

    transform = build_transform(config)

    test_dataset = UBCDataset_Origin(
        csv_path=config.csv_path,
        imgs_path=config.img_folder,
        img_size=config.img_size,
        transforms=transform
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return test_dataloader


def val_model(model, test_dataloader, config):
    model.eval()
    with torch.no_grad():
        with tqdm(test_dataloader, desc="Test") as t:
            for x, y in t:
                x = x.to('cuda', non_blocking=True)
                y = y.to('cuda', non_blocking=True)
                y_p = model(x)

                pred = torch.argmax(y_p, dim=1)

                top_1 = accuracy(y_p, y, topk=(1,))

                top_1_acc = top_1.item() if isinstance(top_1, torch.Tensor) else top_1[0]

                test_timm_acc = '{:.4f}'.format(top_1_acc)

                y = y.to('cpu')
                pred = pred.to('cpu')
                test_balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

                t.set_postfix(test_acc=test_timm_acc, test_BA=test_balanced_accuracy)

                logger.info(f'test:'
                            f'test Accuracy: {test_timm_acc},'
                            f'test Balanced Accuracy: {test_balanced_accuracy}')

if __name__ == "__main__":
    test_dataloader = build_test_dataloader(config)
    model.load_state_dict(torch.load("/root/autodl-tmp/medical_cls/convnext_tiny--2023-11-30--epoch50--ver2.pt"))
    val_model(model, test_dataloader, config)
