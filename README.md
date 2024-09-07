Some dependencies require  python to be below 3.11  
So use conda (or venv if u prefer):
```
conda create -n turbo-O python=3.10
conda activate turbo-O
pip install -r requirements.txt
```

To run the example jupyter notebook, you also need to install matplotlib  

```
conda activate turbo-O
pip install matplotlib
```