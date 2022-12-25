import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from retinanet.dataloader import ImageDirectory, custom_collate
from retinanet.utils import get_logger
from retinanet import model
import json

logger = get_logger(__name__)

dataset = ImageDirectory('test/')

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = model.resnet50(
            num_classes=11,
            pretrained=False,
            conf_threshold=0.7,
        )
weights = 'models/retinanet_resnet50_best.pt'
#ckpt = torch.load(weights, map_location=device)
model.load_state_dict(torch.load(weights, map_location="cpu"))  # Choose whatever GPU device number you want
#model.load_state_dict(ckpt)
model.to(device)
model.eval()


loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )

print(loader)

preds = dict()
for i, (batch, filenames) in tqdm(enumerate(loader), total=len(loader)):
    #print(batch[0])
    with torch.no_grad():
        #img_id, confs, classes, bboxes = model(batch[0].float().cuda())
        img_id, confs, classes, bboxes = model((batch[0].float()).unsqueeze(0))
    img_id = img_id.cpu().numpy().tolist()
    confs = confs.cpu().numpy()
    classes = classes.cpu().numpy()
    bboxes = bboxes.cpu().numpy().astype(np.int32)

    print(classes)
logger.info(f"predictions saved in output.json")