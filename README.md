# canny

full project here:


NOTES FOR PPM FILES

    1) Header must be in this format:
        P6 IMGWIDTH IMGHEIGHT 255
    2) Check PPMs in img/ folder for examples



NOTES FOR CANNY (seq and tbb)

    1) ensure your project directory structure looks like this:

        proj/
            canny.cpp
            Makefile
            img.ppm

        ensure img.ppm is in the same directory as the generated executable

    2) make sure you have gcc/g++ if not already installed, and the Threading Building Blocks library.

        sudo apt-get update
        sudo apt-get install libtbb-dev
        sudo apt-get install gcc g++ 
        
    3) use directions after building with make:

        ./canny MODE PPM ITER
            MODE: 0 for sequential, 1 for TBB, 2 for both
            PPM: 0 for printing only final canny PPM, 1 for printing all PPMs
            ITER: # of iterations



NOTES FOR CANNYCUDA (GPU)

    1) Follow installation guide for CUDA here:
        https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
        https://developer.nvidia.com/cuda-downloads
    
    2) You will need libwb to build the project. Git clone from here:
        https://github.com/abduld/libwb.git
        git@github.com:abduld/libwb.git

    3) build libwb
        cd [path_to_libwb]
        mkdir build
        cd build
        cmake ..
        make
    
    4) ensure directory structure looks like this:

        parent/
            libwb/
                build/
                ...
            proj/
                cannycuda.cu
                CMakeLists.txt
                build/
                    img.ppm
                    ...
                
            
        ensure img.ppm is in the same directory as the generated executable

    5) to build cannycuda

        cd proj/build
        cmake ..
        make
    
