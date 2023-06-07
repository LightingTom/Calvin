# Calvin

Reproduce the paper: [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/pdf/2112.03227.pdf)

The original code: https://github.com/mees/calvin

To run the code, please download the dataset (provided in the above github repo):

```shell
wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
```

Please put the dataset at the same directory as `train.py`.

Or modify the address in `Dataset.py`:

Then, run `train.py`, and the result will be recorded in `calvin.log`.

The result I get is recorded in `result.log`.