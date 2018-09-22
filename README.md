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
CUDA_VISIBLE_DEVICES=1 python main.py multivariate 2 2 2 2
```

# Run all tests
```
CUDA_VISIBLE_DEVICES=1 python run.py
```
