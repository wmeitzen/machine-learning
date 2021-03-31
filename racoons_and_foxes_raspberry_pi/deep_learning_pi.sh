
# echo commands
set -x

# - install pre-request libraries
sudo apt install python-numpy python-scipy python-h5py python-opencv -y # step 1
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl # step 2
sudo pip3 install tensorflow*.whl # step 3
rm tensorflow*.whl
sudo pip3 install keras # step 3
sudo pip3 install opencv-python # step 4
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



