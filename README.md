# Initialization
```
git clone https://github.com/mattgroh/gans6883  
cd gans6883
python3 -m venv env  
. env/bin/activate
pip install -r requirements.txt  
```

# Progress
Multivariate

 | 1k Samples | 10k Samples | 100k Samples | 1M Samples
:---: | :---: | :---: | :---: | :---: |
16 Dim |  |  |  |
32 Dim |  |  |  |
64 Dim |  |  |  |
128 Dim |  |  |  |
256 Dim |  |  |  |
512 Dim |  |  |  |
1024 Dim |  |  |  |

Mixture

| 1k Samples | 10k Samples | 100k Samples
:---: | :---: | :---: | :---: |
16 Dim |  |  |  
32 Dim |  |  |  
64 Dim |  |  |  
128 Dim |  |  |
256 Dim |  |  |  
512 Dim |  |  |  
1024 Dim |  |  |

Circles

1k Samples | 10k Samples | 100k Samples
:---: | :---: | :---: | :---: |
0/20 | 0/20 | 0/20

MNIST (DONE)


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
