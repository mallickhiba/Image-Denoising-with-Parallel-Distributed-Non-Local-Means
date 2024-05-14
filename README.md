# Image-Denoisng-Parallel-Distributed-Non-Local-Means
 
 

wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h 


SEQ
gcc o nlm_seq nlm-seq-time.c -lm -ljpeg
./nlm_seq input_image.jpg
(final)
gcc o <your_output_file> <your_C_File_name.c> -lm -ljpeg
./<your_output_file> <your_input_image.jpg>


MPI
mpicc -o mpiFinal mpiFinal.c -lm -ljpeg
mpirun -np 8  mpiFinal input_image.jpg
(final)
mpicc -o <your_output_file> <your_C_File_name.c> -lm -ljpeg
mpirun -np 8  <your_output_file> <your_input_image.jpg>
