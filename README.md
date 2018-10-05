# Initialization
```
git clone https://github.com/mattgroh/gans6883  
cd gans6883
python3 -m venv env  
. env/bin/activate
pip install -r requirements.txt  
```

# Multivariate

| 1k Samples | 10k Samples | 100k Samples
:---: | :---: | :---: | :---: |
16 Dim | 20/20 | 12/20 | 3/20
32 Dim | 20/20 | 20/20 | 15/20
64 Dim | 20/20 | 18/20 |
128 Dim | 20/20 | 10/20 |
256 Dim | 20/20 | 20/20 |
512 Dim |  17/20| 10/20 |
1024 Dim | 8/20 | 5/20 |

# Mixture (1000 mixtures, only)

 | 1k Samples | 10k Samples | 100k Samples
:---: | :---: | :---: | :---: | :---: |
16 Dim | 20/20 | 9/20 |
32 Dim | 10/20  |  |
64 Dim  |  |  |
128 Dim |  |  |

# Circles

1k Samples | 10k Samples | 100k Samples
:---: | :---: | :---: | :---:
0/20 | 0/20 | 0/20

# MNIST (DONE)

Trials|
:---:|
20/20|


# Run single job
```
CUDA_VISIBLE_DEVICES=3 python3 main.py multivariate 1 32 25 1000
```

# Run batch of jobs
```
# Launches 5 jobs named 1 through 5 on GPU 0 using dataset multivariate, 2 trials
# per job, 32 dimensions, and 10000 samples.

bash job.sh 1 5 0 multivariate 2 32 10000
```

# Run all jobs (this would take years to finish lol)
```
CUDA_VISIBLE_DEVICES=0 python3 run.py
```
