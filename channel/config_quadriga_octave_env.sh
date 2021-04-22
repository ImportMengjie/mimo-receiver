#!/bin/bash

shell_file_dir=$(cd "$(dirname "$0")";pwd)

QuaDRiGa_download_url='https://quadriga-channel-model.de/?download=31747'
QuaDRiGa_path="${shell_file_dir}/QuaDriGa_2020.11.03_v2.4.0.zip"
curl "$QuaDRiGa_download_url" -o "${shell_file_dir}/QuaDriGa_2020.11.03_v2.4.0.zip"

unzip "${QuaDRiGa_path}" -d "${shell_file_dir}"
rm "${shell_file_dir}/quadriga_documentation_v2.4.0-0.pdf" "${shell_file_dir}/QuaDRiGa_License.txt" "${shell_file_dir}/QuaDriGa_2020.11.03_v2.4.0.zip"
rm -rf "${shell_file_dir}/tutorials"

echo "addpath('${shell_file_dir}/quadriga_src')" >> ~/.octaverc
echo "more off" >> ~/.octaverc
