#!/bin/bash


# apt-get install apt-transport-https ca-certificates gnupg curl
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# apt-get update && apt-get install google-cloud-cli



mkdir -p input/goes16/
cd input/goes16/


for i in {198..211}; do  # May

    i=$(printf "%03d" "$i")
    mkdir -p $i
    cd $i 

    for j in {0..24}; do  # 24h
        j=$(printf "%02d" "$j")
        mkdir -p $j
        cd $j

        gsutil -m cp gs://gcp-public-data-goes-16/ABI-L1b-RadF/2023/$i/$j/OR_ABI-L1b-RadF-M6C11_* .
        gsutil -m cp gs://gcp-public-data-goes-16/ABI-L1b-RadF/2023/$i/$j/OR_ABI-L1b-RadF-M6C14_* .
        gsutil -m cp gs://gcp-public-data-goes-16/ABI-L1b-RadF/2023/$i/$j/OR_ABI-L1b-RadF-M6C15_* .

        cd ..
        
#         break 2
    done
    
    cd ..
#     break 2
done
