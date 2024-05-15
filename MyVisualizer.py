import numpy as np
from matplotlib import pyplot as plt


def plot(images, predict, target):
    scale = np.array([images.shape[3], images.shape[2], images.shape[3], images.shape[2]])
    # show the first image and bbox
    img = images[0].permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img)
    bbox = predict[1][0].detach().cpu().numpy()
    cls = predict[0][0].detach().cpu().numpy()
    # dont print 0 class
    bbox = [box for i, box in enumerate(bbox) if cls[i].argmax() != 0]
    if len(bbox) > 0:
        bbox = np.stack(bbox)
        bbox = bbox * scale
        for box in bbox:
            plt.gca().add_patch(
                plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='r', linewidth=1))
    bbox = target[1][0].detach().cpu().numpy()
    cls = target[0][0].detach().cpu().numpy()
    # dont print 0 class
    bbox = [box for i, box in enumerate(bbox) if cls[i] != 0]
    if len(bbox) > 0:
        bbox = np.stack(bbox)
        bbox = bbox * scale
        for box in bbox:
            plt.gca().add_patch(
                plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='b', linewidth=1))
    plt.show()
