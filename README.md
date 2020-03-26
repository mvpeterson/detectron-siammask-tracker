# detectron-siammask-tracker

1. Clone [detectron2 repo](https://github.com/mvpeterson/detectron2) and follow [installation guide](https://github.com/mvpeterson/detectron2/blob/master/INSTALL.md). (I installed pre-built version using pip)

2. Download a model from [model zoo](https://github.com/mvpeterson/detectron2/blob/master/MODEL_ZOO.md). (I used Mask R-CNN X152)

3. Clone [SiamMask repo](https://github.com/mvpeterson/SiamMask) and follow [environment setup guide](https://github.com/mvpeterson/SiamMask#environment-setup)

```
cd SiamMask
export SiamMask=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
``` 
4. run tracking_demo.py
