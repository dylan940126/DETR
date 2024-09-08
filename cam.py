import torch
import torch_tensorrt
import os.path
import psutil
from torchvision import transforms
from pycocotools.coco import COCO
from torch.backends import cudnn
from MyDETR import DETR
import cv2


def detect():
    coco_val = COCO('coco/annotations/instances_val2017.json')
    cats = {x['id']: x['name'] for x in coco_val.loadCats(coco_val.getCatIds())}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
    ])
    # cam = cv2.VideoCapture('https://192.168.1.108:8080/video')
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            break
        frame = transform(frame)
        image = frame.unsqueeze(0).to(device)
        frame = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        pred_cat, pred_bbox = model(image)
        # draw bbox
        wait = 1
        for cat, bbox in zip(pred_cat[0], pred_bbox[0]):
            cat = cat.argmax().item()
            if cat == 0:
                continue
            cat_name = cats[cat]
            if cat_name == 'tie':
                wait = 0
            cx, cy, w, h = bbox * 512
            x, y = cx - w / 2, cy - h / 2
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, cat_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # Parameters
    val_batch_size = 1
    num_workers = 0
    num_classes = 91
    num_queries = 100
    hidden_dim = 128
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0

    # Device
    batt = psutil.sensors_battery()
    if torch.cuda.is_available() and (batt is None or batt.power_plugged is True):
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = DETR(num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries, dropout, False)
    model.to(device).eval()
    path = input('Input model path: ')
    if path != '':
        _, ext = os.path.splitext(path)
        if ext == '.pth':
            print('Loading checkpoint...')
            convert = input('Convert to TensorRT? (y/n): ')
            if convert == 'y':
                save = input('Save to path? (leave blank to not save): ')
            load_checkpoint = torch.load(path, weights_only=True)
            model.load_state_dict(load_checkpoint)
            if convert == 'y':
                print('Converting to TensorRT...')
                inp = torch.ones((1, 3, 512, 512)).to(device)
                model = torch_tensorrt.compile(model, ir="dynamo", inputs=[inp], enabled_precisions={torch.half})
                if save != '':
                    torch.save(model, save)
                    print(f'Saved to {save}')
        elif ext == '.ep':
            print('Loading exported model...')
            model = torch.export.load(path).module()
        else:
            raise ValueError('Invalid model path')
    else:
        raise ValueError('Invalid model path')
    detect()
