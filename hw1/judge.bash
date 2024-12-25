#!/bin/bash

for i in $(seq -w 1 2); do
    hw1-judge -i "$i.txt" 2>&1 | grep -E 'accepted|wrong|runtime error'
done
