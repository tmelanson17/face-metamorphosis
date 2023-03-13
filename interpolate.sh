#!/bin/bash

# Arguments:
#  $1 -- directory of images
#  $2 -- number of frames of interpolation between images (default 50)

sequence_name=$1
frames=$2
if [[ $frames -eq "" ]]
then
    frames=50
fi

output_dir=output_$sequence_name
mkdir -p $output_dir
echo "Creating sequences, outputting to $output_dir ..."
python main.py --images `ls $sequence_name/*`  --frames $frames  --output-dir $output_dir

# List the sequences in the order they were created, to match the python sequence.
echo "Creating the gifs for $sequence_name ..."
for dir in `ls -ctr $output_dir`; do
    gif_name="${sequence_name}_$dir"
    ffmpeg -i ${output_dir}/$dir/mix/mix_%04d.png $gif_name.gif
done

# List the sequences in the order they were created, to match the python sequence.
echo "Combining the gifs into one ..."
rm mylist.txt
touch mylist.txt
for name in `ls -ctr ${sequence_name}_*.gif`; do
    echo "file '${name}'" >> mylist.txt
done

ffmpeg -f concat -i mylist.txt output_${sequence_name}.gif

