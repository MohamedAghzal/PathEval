g++ -std=c++17 -o GeneratePaths GeneratePaths.cpp -I$HOME/ompl_install/include/ompl -L$HOME/ompl_install/lib64 -lompl

echo 'export CPLUS_INCLUDE_PATH=$HOME/boost/include:$CPLUS_INCLUDE_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/boost/lib:$HOME/ompl_install/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

python generate_paths.py $1 $2 $3 $4