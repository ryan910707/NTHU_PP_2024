#!/bin/bash

for i in $(seq -w 1 40); do
    hw1-judge -i "$i.txt" 2>&1 | grep 'accepted'
done
