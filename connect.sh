#!/bin/bash

# Change the IP address (or machine name) with each restart.
# When you have your own username, i.e. after you have created individual user accounts on the EC2 instance,
# make a copy of this script and change the "NAME" field to your own account name.
# Keep the original of this script, since "ec2-user" will still be your group's master account, and has sudo access. 

ADDR=52.37.235.53             # change each time you start your machine to its current dns name or ip address
NAME=ubuntu
LHOST=localhost
SSHKEY=CS280.pem          # change to the name of your private key file

for i in `seq 8888 8900`; do
    FORWARDS[$((2*i))]="-L"
    FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
done
other_ports=( 8080 8081 4040 )
for i in ${other_ports[@]}; do
    FORWARDS[$((2*i))]="-L"
    FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
done

ssh -i ${SSHKEY} -X ${FORWARDS[@]} -l ${NAME} ${ADDR}

