id=$(docker build -q .)
nvidia-docker run -it -p 443:443 -p 8888:8888 -p 6006:60006 -v `pwd`/notebooks:/notebooks $id
