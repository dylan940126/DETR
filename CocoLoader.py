from pycocotools.coco import COCO


class CocoLoader:
    def __init__(self, ann_file):
        self.catIds = None
        self.imgIds = None
        self.coco = COCO(ann_file)
        self.batch_size = None
        print('categories:', ', '.join([cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]))

    def batch(self, batch_size, cat=None):
        self.batch_size = batch_size
        self.catIds = self.coco.getCatIds(cat)
        if cat is None:
            self.imgIds = self.coco.getImgIds()
        else:
            self.imgIds = self.coco.getImgIds(catIds=self.catIds)

        for i in range(0, len(self.imgIds), self.batch_size):
            yield self.imgIds[i:i + self.batch_size]

    def getImgs(self, idx):
        return self.coco.loadImgs(idx)

    def getAnns(self, idx):
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=idx, catIds=self.catIds))

    def __len__(self):
        return len(self.imgIds)
