# Fashion-MNIST Machine Learning Task

Python version for this project is `3.7.5`.<br/>
<br/>
List of used packages and their respective versions:
* `matplotlib==3.1.1`
* `seaborn==0.9.0`
* `pandas==0.25.3`
* `numpy==1.17.4`
* `sklearn==0.21.3`
* `tensorflow==2.0.0` (gpu version)

Project interactive presentation / documentation is `Fashion-MNIST.ipynb` jupyter notebook.<br/>
<br/>
Project modules:
* `data_utils.py` - data preparation and data augmentation code
* `models.py` - keras models specifications (implementations)
* `plot_utils.py` - data visualization and plotting module
* `run_experiments.py` - main script for running experiments
* `train_utils.py` - main function for model training and other training related stuff


Project folders:
* `images` - images of implemented architectures
* `results` - results of experiments, saved in `.json` format
* `weights` - weights of the best models trained, saved in `.h5` format
