import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from YOLO_model import YOLOv1
from dataset import VOCDataset

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

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0
BATCH_SIZE = 1
EPOCHS = 100
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


def main():
    model = YOLOv1(architecture_config, split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    train_dataset = VOCDataset("./dataset/8examples.csv",
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

    for i, data in enumerate(train_loader):
        print(i)


if __name__ == "__main__":
    main()
