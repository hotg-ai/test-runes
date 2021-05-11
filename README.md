

# Mac Run

On Mac `rune run` is unstable so I made a container for us to test in.

## Setup

```
$ docker build util/Dockerfile -t hotg/tester

$ alias runed="sh `pwd`/util/rune.sh" 

$ cd bird_classification

$ rune build Runefile # build as normal

$ runed run ./data/bird_classification.rune  --capability=image:./data/cropped_alcedo_atthis.jpg

```