#docker build . -t hotg/tester
echo "Runing with cmd: $@"
docker run -v `pwd`:/app/data -w /app  -i -t hotg/tester ./rune $@