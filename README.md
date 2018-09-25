# Initialization
```
git clone https://github.com/mattgroh/gans6883  
cd gans6883
python3 -m venv env  
. env/bin/activate
pip install -r requirements.txt  
```

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
