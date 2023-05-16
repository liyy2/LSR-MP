pip install opencv-python-headless
pip install torch==1.9.0+cu111 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
pip install torch-scatter==2.0.8 torch-sparse==0.6.10 torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric==2.0.3
pip install torchtext==0.10.0
pip install pytorch_lightning==1.5.0 #seems not work in v100
pip install -r lightnp_env_requirements.txt
pip install markupsafe==2.0.1
conda install yaml -y
pip install wandb
pip install sympy
pip install torch-ema
pip install ase