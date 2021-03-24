rm -rf oneDNN
git clone https://github.com/oneapi-src/oneDNN.git oneDNN
cd oneDNN
git checkout tags/v2.1.2
mkdir -p build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install/ ..
make -j50
make doc
make install
