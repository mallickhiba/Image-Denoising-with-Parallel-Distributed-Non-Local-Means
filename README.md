# Image-Denoising-Parallel-Distributed-Non-Local-Means
 
Hiba Mallick - 24015 & Muhammad Musab Iqbal - 24495

##HOW TO RUN
To download stb_image.h and stb_image_write.h  libraries, navigate to either the sequential folder or the MPI folder and run:

wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h 


###SEQ
In the sequential folder run:
gcc o <your_output_file> <your_C_File_name.c> -lm -ljpeg
./<your_output_file> <your_input_image.jpg>


###MPI
In the MPI folder run:
mpicc -o <your_output_file> <your_C_File_name.c> -lm -ljpeg
mpirun -np 8  <your_output_file> <your_input_image.jpg>

###CUDA
In the Image_Denoising_with_CUDA.ipynb, run the CUDA setup section, project section and the output viewing section to view your denoised images.
