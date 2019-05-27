# Learning Actor Relation Graphs for Group Activity Recognition

Source code for the following paper([arXiv link](https://arxiv.org/abs/1904.10117)):

        Learning Actor Relation Graphs for Group Activity Recognition
        Jianchao Wu, Limin Wang, Li Wang, Jie Guo, Gangshan Wu
        in CVPR 2019
        
        


## Dependencies

- Python `3.x`
- PyTorch `0.4.1`
- numpy, pickle, scikit-image
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)
- Datasets: [Volleyball](https://github.com/mostafa-saad/deep-activity-rec), [Collective](http://vhosts.eecs.umich.edu/vision//activity-dataset.html)




## Prepare Datasets

1. Download [volleyball](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) or [collective](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip) dataset file.
2. Unzip the dataset file into `data/volleyball` or `data/collective`.




## Get Started

1. Stage1: Fine-tune the model on single frame without using GCN.

    ```shell
    # volleyball dataset
    python scripts/train_volleyball_stage1.py
    
    # collective dataset
    python scripts/train_collective_stage1.py
    ```

2. Stage2: Fix weights of the feature extraction part of network, and train the network with GCN.

    ```shell
    # volleyball dataset
    python scripts/train_volleyball_stage2.py
    
    # collective dataset
    python scripts/train_collective_stage2.py
    ```
    
    You can specify the running arguments in the python files under `scripts/` directory. The meanings of arguments can be found in `config.py`



## Citation

```
@inproceedings{CVPR2019_ARG,
  title = {Learning Actor Relation Graphs for Group Activity Recognition},
  author = {Jianchao Wu and Limin Wang and Li Wang and Jie Guo and Gangshan Wu},
  booktitle = {CVPR},
  year = {2019},
}
```



