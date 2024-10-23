export CONDA_SOURCE=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
export CONDA_INSTALLER=Miniconda3.sh
export SAVE_FOLDER=/tmp/conda_env
export ENV_NAME=conda_env_3_10
export PYTHON_VERSION=3.10

mkdir $SAVE_FOLDER
wget $CONDA_SOURCE -O $SAVE_FOLDER/$CONDA_INSTALLER

# install miniconda
chmod +x $SAVE_FOLDER/$CONDA_INSTALLER
bash $SAVE_FOLDER/$CONDA_INSTALLER -b -p $SAVE_FOLDER/miniconda

# activate conda and create env
export PATH="$PATH:/tmp/conda_env/miniconda/bin"
source activate base
conda info --envs
conda create -n $ENV_NAME python=$PYTHON_VERSION --yes
conda info --envs
conda activate $ENV_NAME
echo $CONDA_PREFIX

# install the packages
conda install -y conda-pack pip
conda install -y torchvision
conda install -y -c conda-forge libstdcxx-ng=12 libstdcxx-ng=12

# pack the env
conda pack -fo $SAVE_FOLDER/$ENV_NAME.tar.gz
mv $SAVE_FOLDER/$ENV_NAME.tar.gz /models/cifar10-preprocessing/$ENV_NAME.tar.gz
conda deactivate

# optionally, cleanup
# # rm -r $SAVE_FOLDER