#!/bin/bash

print_help(){
    echo "$0 - Unzip multiple .zip files"
    echo ""
    echo "Usage: $0 [-p|h|r] [FILENAME.zip...]"
    echo ""
    echo "options:"
    echo "-h,           show brief help"
    echo "-p,           process in parallel"
    exit 0
}

if [ $# -eq 0 ]; then
    echo "No arguments to command:"
    print_help
fi

while getopts ':hpr' flag; do
    case "${flag}" in
        h) 
            shift 
            print_help
            exit ;;
        p)
            shift
            parallel_process=true ;;
        \?)
            echo "Error: Invalid Option: ${OPTARG}"
            exit ;;
    esac
done

jobs=8 # default number of jobs

test_and_unzip() {
    file="${1}"
    if unzip -q -t "${file}" 
    then
        unzip -qu "${file}"
    else
        echo "$File does not pass CRC check"
        file_basename=$(basename "${file}")
        mv "${file}" "${file_basename}.broken"
        exit 1
    fi
}
export -f test_and_unzip

if [[ "${parallel_process}" != true ]]
then
    for file in "${@}"
    do
        test_and_unzip "${file}" 
    done
else
    parallel --jobs "${jobs}" "test_and_unzip {}" ::: "${@}"
fi
