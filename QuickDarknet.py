#coding=utf-8
if __name__ == "__main__": 
    import darknet as dn # darknet is the file in the same path
else:
    from darknet import darknet as dn # darknet is the folder in the ~/anaconda3/lib/python3.6/site-packages/darknet
import os
import sys
dir = os.path.dirname(dn.__file__)

class QuickDarknet:
    def __init__(self):
        dn.set_gpu(0)
        self.net = dn.load_net(str.encode(os.path.join(dir, "cfg/yolov3.cfg")), str.encode(os.path.join(dir, "weights/yolov3.weights")), 0)
        self.meta = dn.load_meta(str.encode(os.path.join(dir, "cfg/coco.data")))

    def detect(self, image_path: str, keyword:str="", threshold=0.7):
        r = dn.detect(self.net, self.meta, str.encode(image_path))
        r = [[key, prob, [int(x) for x in pos]] for key, prob, pos in r]
        if keyword:
            r = [[key, prob, pos] for key, prob, pos in r if key == keyword and prob > threshold]
        return r

if __name__ == "__main__":
    qd = QuickDarknet()
    r = qd.detect("data/dog.jpg")
    print(r)

