CC=clang
CFLAGS=-Wall -g
SDLStuff=-I./include/SDL2 -L./lib -lSDL2main -lSDL2

all: xor add convertPngMat upscaleImg

xor: xor.c nn.c
	$(CC) $(CFLAGS) -o bin/$@ $^ -lm

add: add.c nn.c plot.c trainer.c
	$(CC) $(CFLAGS) -o bin/$@ $^ -lm $(SDLStuff)

convertPngMat: convertPngMat.c nn.c
	$(CC) $(CFLAGS) -o bin/$@ $^ -lm

upscaleImg: upscaleImg.c nn.c nnFile.c plot.c trainer.c
	$(CC) $(CFLAGS) -o bin/$@ $^ -lm $(SDLStuff) -lSDL2_ttf
