SHELL := /bin/bash

.PHONY: all
all:
	pip install runcython
	makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

hungarian: hungarian/hungarian.so

hungarian/hungarian.so:
	cd hungarian && \
	TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
	TF_LIB=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') && \
	if [ `uname` == Darwin ];\
	then g++ -std=c++11 -shared hungarian.cc -o hungarian.so -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -I $$TF_INC -undefined dynamic_lookup -L$$TF_LIB -ltensorflow_framework;\
	else g++ -std=c++11 -shared hungarian.cc -o hungarian.so -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -I  $$TF_INC -L$$TF_LIB -ltensorflow_framework; fi
