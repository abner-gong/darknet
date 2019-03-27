#coding=utf-8
if __name__ == "__main__":
    import darknet as dn
else:
    from darknet import darknet as dn
import os
import sys
dir = os.path.dirname(dn.__file__)
print("darknet path", dir)

class QuickDarknet:
    def __init__(self, tiny=False):
        try: # keep only one instance design to avoid "out of memory"
            self.meta = QuickDarknet.meta 
            self.net = QuickDarknet.net
            return
        except: # if no instance then go ahead
            pass
        dn.set_gpu(0)
        if tiny: # tiny-weights would be faster but not precise enough
            self.net = dn.load_net(str.encode(os.path.join(dir, "cfg/yolov3-tiny.cfg")), str.encode(os.path.join(dir, "weights/yolov3-tiny.weights")), 0)
        else:
            self.net = dn.load_net(str.encode(os.path.join(dir, "cfg/yolov3.cfg")),
                                   str.encode(os.path.join(dir, "weights/yolov3.weights")), 0)
        self.meta = dn.load_meta(str.encode(os.path.join(dir, "cfg/coco.data")))
        QuickDarknet.meta = self.meta # save the instance to class static member
        QuickDarknet.net = self.net

    def detect(self, image_path: str, object_name:str="", threshold=0):
        r = dn.detect(self.net, self.meta, str.encode(image_path))
        r = [[str(key, encoding="utf-8"), prob, [int(x) for x in pos]] for key, prob, pos in r] #str.encode() can't work here
        for key, prob, pos in r:
            print(key, object_name, key == object_name)
        if object_name:
            r = [item for item in r if item[0] == object_name]
        if threshold:
            r = [item for item in r if item[1] > threshold]
        return r
        # ========notice==========
        # r is a list of (name, prob, [center of x, center of y, width, height])
        # however, center of x - width//2 maybe below zero, so you should handle r like this:
        #
        #     for key, prob, pos in r:
        #          x, y, w, h = pos
        #          h1, w1, _ = im.shape
        #          im2 = im[max(0, y-h//2):min(y+h//2, h1-1), max(x-w//2,0):min(x+w//2, w1-1)]


if __name__ == "__main__":
    qd = QuickDarknet()
    r = qd.detect(os.path.join(dir, "data/dog.jpg"))
    print(r)


