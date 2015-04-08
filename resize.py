#!/home/hendrik/work/ndsb/bin/python

"""
indir: root directory of where the training images are located
outdir: root directory of where the output training images will be placed
command: the command that will be applied to each image before it is moved to outdir
allfile: the file where all processed files will be written as "/path/to/file label"
headerfile: file with ordered class labels as "image,label1,label2,label3,..."
"""

import sys
import subprocess

from os import walk, makedirs, listdir
from os.path import join, split, exists, splitext


"""
test data structure:
    test_dir
        img1
        img2
"""


def resize_test(indir, outdir, command, convert_to_png=False):
    allfiles = listdir(indir)
    for fname in allfiles:
        infile = join(indir, fname)
        outfile = join(outdir, fname)

        resize_image(infile, outfile, command, convert_to_png)


"""
training data structure:
train_dir
    class1_dir
        class1_img1
        class1_img2
    class2_dir
        class2_img1
"""


def resize_training(indir, outdir, command, allfile, headerfile, convert_to_png=False):
    with open(allfile, 'w') as afile, \
            open(headerfile, 'w') as hfile:
        hfile.write('image')
        for label, (root, dirs, files) in enumerate(walk(indir)):
            dirs.sort()   # it's easier if the classes are sorted alphabetically
            if root == indir:
                continue
            _, parent = split(root)
            hfile.write(',%s' % (parent))
            print('processing directory %s' % (parent))
            new_dir = join(outdir, parent)
            if not exists(new_dir):
                makedirs(new_dir)

            for fname in files:
                infile = join(root, fname)
                outfile = join(new_dir, fname)

                afile.write('%s %d\n' % (outfile, label - 1))  # the first directory is the root

                resize_image(infile, outfile, command, convert_to_png)


def resize_image(infile, outfile, command, convert_to_png=False):
    if convert_to_png:
        outfile = splitext(outfile)[0] + '.png'
    new_command = command.replace('$name_in', infile)
    new_command = new_command.replace('$name_out', outfile)

    retcode = subprocess.call([new_command], shell=True)
    if retcode != 0:
        print('error resizing image %s' % (infile))

    return outfile


if __name__ == '__main__':
    root = '/home/hendrik/work/ndsb/data'
    command = 'convert $name_in -resize 95x95 -gravity center -background white -extent 95x95 $name_out'
    # command = 'convert $name_in -resize 48x48! $name_out'  # the ! tells imagemagick to ignore aspect ratio

    if len(sys.argv) != 2:
        print('usage:\n ./resize.py train\n ./resize.py test')
        exit(0)

    arg = sys.argv[1].lower()
    if arg == 'train':
        indir = join(root, 'train')
        outdir = join(root, 'train-resized-95')
        allfile = join(root, 'kaggle-all-95.txt')
        headerfile = join(root, 'kaggle-header-95.txt')

        print('resizing the training set:')
        print(' the input directory is %s' % (indir))
        print(' the output directory is %s' % (outdir))
        print(' the filenames will be written to %s' % (allfile))
        print(' the header file will be written to %s' % (headerfile))
        print(' the resizing command is %s' % (command))

        resize_training(indir, outdir, command, allfile, headerfile, convert_to_png=False)
    elif arg == 'test':
        indir = join(root, 'test')
        outdir = join(root, 'test-resized-noaspect')

        print('resizing the test set:')
        print(' the input directory is %s' % (indir))
        print(' the output directory is %s' % (outdir))

        resize_test(indir, outdir, command)
