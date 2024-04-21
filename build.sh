#!/bin/sh

hasBear=$(command -v bear);
prefix=""

# Some systems (mine) need a program called Bear to generate a compile_commands.json file so that autocomplete, autoimport and linting work correctly.
# Since not everyone needs this or has Bear installed this is my way of making a build script that will work for both parties.
if [ ! -z $hasBear ];
then
    prefix="bear --";
fi;

$prefix clang -g -Wall -Wextra -o bin/xor xor.c -lm
$prefix clang -g -Wall -Wextra -o bin/add add.c plot.c trainer.c -lm -I./include/SDL2 -L./lib -lSDL2main -lSDL2
$prefix clang -g -Wall -Wextra -o bin/convertPngMat convertPngMat.c -lm
