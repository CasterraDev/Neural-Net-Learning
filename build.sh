#!/bin/sh

hasBear=$(command -v bear);
clangCommand="clang -Wall -Wextra -o ml *.c -lm"

if [ -z $hasBear ];
then
    $clangCommand;
else
    bear -- $clangCommand;
fi;
