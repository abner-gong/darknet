# Usage 用法
```
from darknet.QuickDarknet import QuickDarknet
qd = QuickDarknet()  # load the weights and meta data for once 载入一次
for image_uri in tqdm(image_list):
    r = qd.detect(image_uri, "person", threshold) # detect for many times 运行多次
```

# Install 安装
## 1. The same as official version 与官方版本相同
```
git clone https://github.com/pjreddie/darknet
cd darknet
make
mv libdarknet.so /lib 
mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
#tiny version: wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

## 2. Move this folder into your site-packages directory 将这个文件夹移动到你的site-packages目录中
```
mv darknet ~/anaconda3/lib/python3.6/site-packages/
```
If you don't know your packages path, use `print(sys.path)` to see it 如果你不知道包目录在哪，用`print(sys.path)`查找

## 3. Change the line`names=xxxx` inside `cfg/coco.data` to correct absolute path of directory 
将`cfg/coco.data`这个文件中的`names=xxxx`这行的目录改为正确的绝对目录

