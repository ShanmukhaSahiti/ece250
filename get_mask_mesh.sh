#! /bin/bash

# conda environment with tensorflow and python 3
echo "Activating tf-python3 environment..."
source $HOME/miniconda3/bin/activate tf-python3
echo "Current Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda Python: $HOME/miniconda3/envs/tf-python3/bin/python"
cd Mask_RCNN
$HOME/miniconda3/envs/tf-python3/bin/python get_mask_example.py
cd ..
echo "Deactivating tf-python3 environment..."
$HOME/miniconda3/bin/conda deactivate

# conda environment with tensorflow and python 2
echo "Activating tf-py2 environment..."
source $HOME/miniconda3/bin/activate tf-py2
echo "Current Conda environment: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"
cd human\ mesh\ recovery
python get_hmr_example.py
cd ..
echo "Deactivating tf-py2 environment..."
$HOME/miniconda3/bin/conda deactivate