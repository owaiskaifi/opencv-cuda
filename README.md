# opencv-cuda


# Build OPENCV VERSION 4.10.0 on: Ubuntu 22.04,  GCC-10  NVIDIA-SMI 565.77                 Driver Version: 565.77         CUDA Version: 12.7  

**Create a Project Folder: `~/project`**
```bash
mkdir ~/project  
cd ~/project/    
```

**Pre-checks Before Proceeding**
```bash
nvidia-smi -L
nvcc --version  
python3 --version
```

# Install NVIDIA Drivers and Utilities if `nvidia-smi -L` Fails
```bash
sudo apt install nvidia-driver-550 nvidia-utils-550
# A system reboot will be necessary if you install the GPU driver.
sudo reboot
# After system reboot, try again
nvidia-smi -L
```

# Install CUDA Toolkit if `nvcc --version` Fails
```bash
sudo apt install nvidia-cuda-toolkit
```

# Install cuDNN 9.1.1:

**Network Install** (recomended)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt -y install cudnn9-cuda-12
```
**Local Install** 
```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2004-9.1.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.1.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt -y install cudnn9-cuda-12
```
# Install Video Codec SDK 12.2 Header Files  (Install if required)
**You will need an Nvidia account to download these files from: [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download).**

```bash
unzip Video_Codec_Interface_12.2.72.zip
sudo cp ~/project/Video_Codec_Interface_12.2.72/Interface/*.h /usr/include/
```
**Note: The following files are no longer needed and can be safely removed:**
* `Video_Codec_Interface_12.2.72`
* `Video_Codec_Interface_12.2.72.zip`  
* `cudnn-local-repo-ubuntu2004-9.1.1_1.0-1_amd64.deb`
```bash
rm -rf ~/project/Video_Codec_Interface_12.2.72 ~/project/Video_Codec_Interface_12.2.72.zip ~/project/cudnn-local-repo-ubuntu2004-9.1.1_1.0-1_amd64.deb
```

# Install CMake and Build Essentials 
**If you don't have gcc-10 and g++-10 installed, the compilation will fail.**
```bash
sudo apt install cmake cmake-data build-essential 
sudo apt install gcc-10 g++-10 
```

# Clone OpenCV and opencv_contrib Repositories
```bash
git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib
```

# Install Dependencies:
```bash
sudo apt install libopenblas-dev libopenblas-base libatlas-base-dev liblapacke-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libv4l-dev v4l-utils
sudo apt install libxvidcore-dev libx264-dev
sudo apt install libgtk-3-dev
sudo apt install protobuf-compiler
sudo apt install python3-dev python3-venv python3-numpy python3-wheel python3-setuptools
sudo apt install tesseract-ocr
```

# Create a Python Environment and Install the Necessary Libraries.
```bash
conda create -n opencv-cuda python=3.11
conda activate opencv-cuda
pip install wheel numpy tesseract

```

# Pre-checks Before Proceeding
```bash
ldconfig -p | grep libopenblas
ldconfig -p | grep libatlas
ldconfig -p | grep liblapacke
```

# Create a Build Folder in the OpenCV Directory
```bash
mkdir ~/project/opencv/build
cd ~/project/opencv/build
```

# Generate configuration for make 
**Note replace the CUDA_ARCH_BIN with your actual compute gpu:** `nvidia-smi --query-gpu=compute_cap --format=csv`
```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export OPENCV_VERSION=4.10.0

cmake \
-D CMAKE_BUILD_TYPE=Release \
-D OPENCV_VERSION=${OPENCV_VERSION} \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=1 \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=8.6 \
-D Atlas_CBLAS_INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/ \
-D OpenBLAS_LIB=/usr/lib/x86_64-linux-gnu/ \
-D OPENCV_EXTRA_MODULES_PATH=/home/owais/owais/personal/opencv-cuda/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=/home/owais/miniconda3/envs/opencv-cuda/bin/python \
-D PYTHON_LIBRARIES=/home/owais/miniconda3/envs/opencv-cuda/lib/python3.11/site-packages \
-D BUILD_opencv_python3=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON ..
```

# Handling lapacke.h File Issues (not needed mostly)
**If there are issues related to lapacke.h, you may need to create a new folder and copy the relevant headers:**
```bash
sudo mkdir /usr/include/openblas
sudo cp /usr/include/lapacke.h /usr/include/openblas/
sudo cp /usr/include/lapacke_mangling.h /usr/include/openblas/
sudo cp /usr/include/lapacke_config.h /usr/include/openblas/
sudo cp /usr/include/lapacke_utils.h  /usr/include/openblas/
sudo cp /usr/include/x86_64-linux-gnu/cblas*.h /usr/include/
rm -rf ~/project/opencv/build/*
# Regenerate configuration for make
```

# Compile and Install it
```bash
make -j$(nproc)
sudo make install
```
# Final Step: 
**Either Link CV2 to the Environment or Use the WHL to Install**

**To link CV2 to the environment:**
```bash
ln -s /usr/local/lib/python3.12/site-packages/cv2 \
  ~/project/.env/lib/python3.12/site-packages/cv2
```

**To create and install the WHL file:** (recomended)
```bash
# To create and install the WHL file:
cd ~/project/opencv/build/python_loader
python3 setup.py bdist_wheel
# Activate the virtual environment
source ~/project/.env/bin/activate
# Check if there is any wheel
ls ~/project/opencv/build/python_loader/dist/
# Install the generated wheel file
pip install dist/opencv-4.10.0-py3-none-any.whl
```
# install gcc in conda 

```
conda install -c conda-forge gcc=12.1.0                                                                 
```
# Check if CUDA is Enabled
```bash
python -c "import cv2; print(f'Cuda Devices: {cv2.cuda.getCudaEnabledDeviceCount()}'); print('OpenCV version:', cv2.__version__);"
```
