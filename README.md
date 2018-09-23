# Initialization
```
git clone https://github.com/mattgroh/gans6883  
cd gans6883
python3 -m venv env  
. env/bin/activate
pip install -r requirements.txt  
```

# Run single test
```
CUDA_VISIBLE_DEVICES=3 python3 main.py multivariate 1 32 25 1000
```

# Run all tests
```
CUDA_VISIBLE_DEVICES=0 python3 run.py
```
