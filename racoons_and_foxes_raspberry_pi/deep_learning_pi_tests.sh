
# echo commands
set -x

# - install pre-request libraries
#sudo apt install python-numpy python-scipy python-h5py python-opencv -y # step 1

#sudo pip3 install opencv-python # step 4

# - Install Tensorflow
#sudo apt install libatlas-base-dev -y # - may not be necessary - returns "libatlas-base-dev is already at the newest version"
#pip install --no-cache-dir tensorflow

# seems to work on rpi 3
#wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl # step 2
#sudo pip3 install tensorflow*.whl # step 3

# the latest version of keras needs tensorflow 2.2, but the latest version of tensorflow
# for the rpi is 1.14.0. So, install tensorflow 1.9.0 instead.
#pip install --no-cache-dir tensorflow==1.9.0
#pip install --no-cache-dir tensorflow==2.2.0 # - no such version in normal repos
#pip install keras --no-cache-dir --no-deps


# - install Keras
#pip install keras==2.1.5 --no-cache-dir --no-deps
#pip install keras --no-cache-dir --no-deps #

#wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
# - https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp35-none-linux_armv7l.whl
#wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp35-none-linux_armv7l.whl -O tensorflow.whl # "tensorflow.whl is not a valid wheel filename"
#wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp35-none-linux_armv7l.whl # "not a supported wheel on this platform"
#rm tensorflow*.whl
#mv tensorflow.whl tensorflow-2.4.0-cp35-none-linux_armv7l.whl
#sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
#sudo pip3 install tensorflow-2.4.0-cp35-none-linux_armv7l.whl

#sudo pip3 install keras # step 3

# begin step 5
sudo pip3 uninstall mock -y
sudo pip3 install mock
sudo apt install libblas-dev
sudo apt install liblapack-dev
sudo apt install python3-dev 
sudo apt install libatlas-base-dev -y
sudo apt install python3-setuptools
sudo apt install python3-scipy -y
sudo apt install python3-h5py
# end step 5

pip3 install numpy --upgrade # step 6
