#! /bin/bash
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip -o ngrok-stable-linux-amd64.zip
pip install --quiet pyngrok
pip install --no-dependencies --quiet protobuf==3.20.*  
pip install --no-dependencies --quiet validators
ngrok authtoken "${NGROK_TOKEN}" 
