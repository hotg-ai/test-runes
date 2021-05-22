# Example Runes

[![Continuous integration](https://github.com/hotg-ai/test-runes/actions/workflows/main.yml/badge.svg)](https://github.com/hotg-ai/test-runes/actions/workflows/main.yml)

## Mac Run

On Mac `rune run` is unstable so I made a container for us to test in.

### Setup

```
$ docker build util/Dockerfile -t hotg/tester

$ alias runed="sh `pwd`/util/rune.sh"

$ cd bird_classification

$ rune build Runefile # build as normal

$ runed run ./data/bird_classification.rune  --capability=image:./data/assets/cropped_alcedo_atthis.jpg

```

## Building and Testing Runes

Set aliases for building and testing runes from project's root directory. **Run the scripts inside the subdirectories (`image/`, `audio/`)**

### Build

```
$ alias runeb="sh `pwd`/util/build.sh"

$ cd image/

$ runeb
```

### Test

##### Image

```console
$ alias runei="sh `pwd`/util/test_image.sh"

$ cd image/

$ runei
```

##### Audio

```console
$ alias runea="sh `pwd`/util/test_audio.sh"

$ cd audio/

$ runea
```
