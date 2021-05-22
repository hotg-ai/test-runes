#docker build . -t hotg/tester
echo "Runing with cmd: $@"
echo ""
# docker run -v `pwd`:/app/data -w /app  -i -t hotg/tester ./rune $@
docker run -v `pwd`:`pwd` -w `pwd` -i -t hotg/tester /app/rune $@