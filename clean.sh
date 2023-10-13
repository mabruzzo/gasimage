#!/usr/bin/env bash

# script to cleanup files created by cython


# path to the directory containing this script
# -> from https://stackoverflow.com/a/630387
LOCAL_DIR_PATH="$(dirname -- "${BASH_SOURCE[0]}")"

# iterate over all paths files in the gasimage directory that end in pyx
# and then delete the .c or .cpp files that are produced by cython
find ${LOCAL_DIR_PATH}/gasimage -name '*.pyx' -type f | while read line; do
    prefix=${line::-4} # clip off the .pyx suffix
    # the -f flag silences warnings if the file doesn't exist
    rm -f "${prefix}.c" "${prefix}.cpp"
done

# delete any files that end in .so
find ${LOCAL_DIR_PATH}/gasimage -name '*.so' -type f -print0 | xargs -0 /bin/rm -f

rm -rf "${LOCAL_DIR_PATH}/build" "${LOCAL_DIR_PATH}/gasimage.egg-info"
