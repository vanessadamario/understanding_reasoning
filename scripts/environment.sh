conda create -n vqa python=3.8
conda activate vqa
pip install PTable --progress-bar off
pip install opencv-python --progress-bar off
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U ray==1.1.0 --progress-bar off
pip install nvgpu --progress-bar off
pip install reprint --progress-bar off
pip install plotly dash dash-core-components dash_html_components dash_table --progress-bar off
conda install -y -q scikit-image
conda install -y -q -c anaconda cython
conda install -y -q jupyter
conda install -y -q numba
conda install -y -q tqdm
conda install -y -q h5py
