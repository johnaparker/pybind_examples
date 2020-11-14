play.cpython-37m-darwin.so: play.cpp 
	g++ -O3 -Wall -shared -std=c++11 -fPIC -Wl,-undefined,dynamic_lookup `python -m pybind11 --includes` play.cpp -o play`python3-config --extension-suffix` -Ipybind11/include:/opt/miniconda3/include/python3.7m
