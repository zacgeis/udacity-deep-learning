# container id below comes from docker build . in docker-sandbox dir
id=$(docker build -q .)
docker run -it -p 8888:8888 -p 6006:60006 -v `pwd`/jnotebooks:/jnotebooks $id

