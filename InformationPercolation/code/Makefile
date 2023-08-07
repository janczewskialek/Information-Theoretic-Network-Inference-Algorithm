all: lib cpp

lib:
	cd FX_infoperc && rm -f fastvecm.cpython-* && python setup.py build_ext --inplace clean --all && cd ..

cpp:
	make -C FX_infoperc/code_cpp