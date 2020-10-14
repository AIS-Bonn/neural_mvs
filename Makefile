all:
	echo "Building NeuralMVS"
	python3 -m pip install -v --user --editable ./ 

clean:
	python3 -m pip uninstall neuralmvs
	rm -rf build *.egg-info build neuralmvs*.so libneuralmvs_cpp.so

        


        
        