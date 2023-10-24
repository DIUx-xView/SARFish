#!/usr/bin/bash

print_help(){
    echo "$0 - Create the python virtual environment for this repo"
    echo ""
    echo "Usage: $0"
    echo ""
    echo "options:"
    echo "-h,           show brief help"
    echo "-p,           set the PIP_CACHE_DIR to store pip cache files"
    echo "-v,           set the path to create the virtual environment"
    echo "-r,           set the path to the pip package requirements file"
    exit 0
}

if [ $# -eq 0 ]; then
    echo "No arguments to command:"
    print_help
fi

while getopts ':hp:v:r:' flag; do
    case "${flag}" in
        h) 
            shift
            print_help
            exit;; #double semicolon indicates the end of an alternative
        p)
            pip_cache_path="${OPTARG}";;
        v)
            virtual_environment_path="${OPTARG}";;
        r)
            requirements_file="${OPTARG}";;
        \?)
            echo "Error: Invalid Option"
            exit;;
    esac
done

if  [[ ! -z "${pip_cache_path}" ]]
then
    echo "pip_cache_path: ${pip_cache_path}"
    export PIP_CACHE_DIR="${pip_cache_path}"
fi

# required arguments
if [[ -z "${virtual_environment_path}" ]]
then
    "missing required argument: -v"
    exit 1
elif [[ -z "${requirements_file}" ]]
then
    "missing required argument: -r"
    exit 1
fi

get_closest_available_pygdal_version_to_system_install() {
    installed_gdal_version=$(
        gdalinfo --version | awk 'BEGIN{FS=",? +"}{print$2}'
    )
    pypi_pygdal_versions_json=$(curl -X GET https://pypi.org/pypi/pygdal/json)
    closest_available_pygdal_version_to_system_install=$(
        jq -r --arg installed_gdal_version "$installed_gdal_version" \
            'last(
                .["releases"]   | to_entries 
                                | .[] 
                                | select(.key | test($installed_gdal_version)) 
                                | [.]
            ) | .[] | .key' <<< "${pypi_pygdal_versions_json}"
    )
    echo "${closest_available_pygdal_version_to_system_install}"
}

echo "virtual_environment_path: ${virtual_environment_path}"
python3 -m venv "${virtual_environment_path}"
source "${virtual_environment_path}/bin/activate"

pygdal_version=$(get_closest_available_pygdal_version_to_system_install)
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r "${requirements_file}"
python3 -m pip install pygdal=="${pygdal_version}"
