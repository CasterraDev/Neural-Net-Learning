#!/bin/sh

hasBear=$(command -v bear);
prefix=""

# Some systems (mine) need a program called Bear to generate a compile_commands.json file so that autocomplete, autoimport and linting work correctly.
# Since not everyone needs this or has Bear installed this is my way of making a build script that will work for both parties.
if [ ! -z $hasBear ];
then
    prefix="bear --";
fi;

$prefix clang -Wall -Wextra -o xor xor.c -lm
$prefix clang -Wall -Wextra -o add add.c -lm
