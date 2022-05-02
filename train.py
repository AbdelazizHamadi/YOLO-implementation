import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from YOLO_model import YOLOv1
from dataset import VOCDataset

from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
)

#from utils.mean_avg_precision import mean_average_precision

#from utils.utils_me import (
#    get_bboxes,
#    load_checkpoint
#)

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# hyper parametres
architecture_config = [

    # tuple (conv layer) : (kernel size, num_filters, stride, padding)
    (7, 64, 2, 3),
    # M : maxpool layer (2x2 - stride 2) as YOLO paper
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",

    # list : [tuple, tuple, num_repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0
BATCH_SIZE = 4
EPOCHS = 500
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.path.tar"

IMG_DIR = "./dataset/images"
LABEL_DIR = "./dataset/labels"


class Compose(object):

    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, img, bboxes):
        for t in self.transforms:
            # transform only on images if for boxes too --> t(bboxes)
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        #model.cuda()
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YOLOv1(architecture_config, split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("./dataset/8examples.csv",
                               transform=transform,
                               img_dir=IMG_DIR, label_dir=LABEL_DIR)

    test_dataset = VOCDataset("./dataset/8examples.csv",
                              transform=transform,
                              img_dir=IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"train mAP:{mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    main()
