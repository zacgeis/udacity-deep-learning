cp Dockerfile.template Dockerfile
echo "temp dockerfile created."
if command -v nvidia-docker 2>/dev/null; then
  echo "nvidia-docker found."
  echo "using latest-gpu-py3 and nvidia-docker."
  sed -i 's/TENSORFLOW_SOURCE_REPLACE/gcr.io\/tensorflow\/tensorflow:latest-gpu-py3/g' Dockerfile
  docker_command=nvidia-docker
else
  echo "nvidia-docker not found."
  echo "using latest-py3 and docker."
  sed -i 's/TENSORFLOW_SOURCE_REPLACE/gcr.io\/tensorflow\/tensorflow:latest-py3/g' Dockerfile
  docker_command=docker
fi
echo "temp dockerfile values rendered."
echo "building..."
id=`docker build -q .`
rm Dockerfile
echo "image build, removing temp dockerfile."
echo "starting jupyter on port 443."
$docker_command run -it -p 443:443 -p 8888:8888 -p 6006:6006 -v `pwd`/notebooks:/notebooks $id
