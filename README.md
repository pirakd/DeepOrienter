# DeepOrienter
### Deep orienter of protein-protein interaction networks
This is the official repo of the paper "D'or: Deep orienter of protein-protein interaction networks"
### Getting started
This project is 100% implemented in python. Authors used python 3.8 but it should run with similar versions as well.
To instaell required packages run:
``` 
pip install path/to/project/requirements.txt
```

### Whats inside
 - **Example input files:** Can be found in [input](input) folder. contains:
    - Directed interactions: A subset of directed interactions used in the paper (KPI's)
    - Priors: A subset of cause effect pairs used in the paper (AML patiens)
    - PPI network: The Full human ANAT network
    - A trained model: Trained using 5 AML patients
    - Membrane receptors and transcription factors used to evaluate Vinayagam's method results.
- **Deep learning model**: found in [deep_learning/models.py](deep_learning/models.py)
- **Example of a parameter configuration**: in [presets.py](presets.py)
- **Train script** example in [scripts/main.py](scripts/main.py) and **model load and inference** example script in [scripts/inference.py](scripts/inference.py)
- **Implementations of two previous methods** mentioned in the paper: [D2D.py](D2D.py) and [Vinayagam.py](Vinayagam.py)


### Citation 


### License
DeepOrienter is MIT licensed, as found in the [LICENSE](LICENSE) file.
