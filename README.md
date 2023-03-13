Libraries covered by the cs6475 conda environment:
- opencv
- numpy
- math
- collections
- os
- argparse

Although most of the code dependencies come with the default cs6475 environment, the dlib library is a 3rd party dependency that will need to be installed.

Since it is not part of the standard conda packages, I will include the install command here:

conda install -c menpo dlib

Additionally, although not a code dependency, the following files are required (and included) in the main directory:
- facial-regions.txt 
- haarcascade_frontalface_alt.xml
- shape_predictor_68_face_landmarks.dat
The first 2 are included. In order to get the second, download the file from <https://drive.google.com/open?id=15J8wQoGapUu0SsadKDgb0fEvDbDr30G4> and put it in the main directory. This was done to reduce the size of resources.zip.

In order to run the code, you run the bash script as follows:

./interpolate.sh <image directory name> <number of frames>

For my particular project, I ran the following:

./interpolate.sh family 50

(for the expressions side project, I changed the directory to faces)

There are two input image directories, family and faces. 

The "family" directory is the main project scope. It has 4 images, each one transitioning to the next consecutive image:
- mom.jpg
- dad.jpg
- granddad.jpg
- grandmom.jpg

The "faces" directory is an extension I used to animate facial expressions. Because the images are large, I also uploaded them online: 
https://drive.google.com/open?id=1mZ7gxWtQ6kSjz1_wBmkUjvwujcQpQewK 
It contains 4 images, each one transitioning to the next consecutive image:
- confused.png
- smile.png
- surprised.png
- wince.png

Output for both the family pictures and the expressions pictures can be found here: https://drive.google.com/open?id=1o4x2ol_5228aUnM_3hqIMswA7cAISs4M

The drive directory is organized as follows:

- output_family.gif : The combined GIF of all 4 family picture transitions
- output_family : The directory containing each individual transition GIF
- output_faces.gif : The combined GIF of all 4 expression transitions
- output_faces : The directory containing each individual transition GIF
