DIR=oneDNN
rm -rf $DIR
git clone https://github.com/oneapi-src/oneDNN.git
cd $DIR
git checkout tags/v1.7
mkdir -p build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install/ ..
make -j50
make doc
make install
