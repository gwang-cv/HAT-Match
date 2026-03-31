# HAT-Match



### HAT-Match: Graph Transformer with Hybrid Attention for Two-View Correspondence Pruning

**ECAI 2025**

![](img/overview.png)

## Requirements

Please use Python 3.7 and Pytorch 1.13. 

Other dependencies should be easily installed through pip or conda.

```bash
pip install -r core/requirements.txt
```

## Train

**Train model on outdoor (yfcc100m) scene**

```bash
python main.py --data_tr=yfcc-sift-2000-train.hdf5 --data_va=data_dump/yfcc-sift-2000-val.hdf5  --log_base=../model/yfcc_sift --gpu_id=0 
```

**Train model on indoor (sun3d) scene**

```bash
python main.py --data_tr=sun3d-sift-2000-train.hdf5 --data_va=sun3d-sift-2000-val.hdf5  --log_base=../model/sun3d_sift --gpu_id=0 
```

## Test

**Test pretrained model on outdoor (yfcc100m) scene**

```bash
python main.py --run_mode=test --data_te=yfcc-sift-2000-test.hdf5  --model_path=../model/yfcc_sift/train/ --res_path=../model/yfcc_sift/test/ --gpu_id=0 
```

**Test pretrained models on indoor (sun3d) scene**

```bash
python main.py --run_mode=test --data_te=sun3d-sift-2000-test.hdf5  --model_path=../model/sun3d_sift/train/ --res_path=../model/sun3d_sift/test/ --gpu_id=0
```

## Results

![](img/res.png)

** --/--  without/with RANSAC**

## Acknowledgement

OANet/CLNet/BCLNet, etc.

