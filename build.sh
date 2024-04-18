#!/bin/sh

hasBear=$(command -v bear);
prefix=""

if [ ! -z $hasBear ];
then
    prefix="bear --";
fi;

$prefix clang -Wall -Wextra -o xor xor.c -lm
