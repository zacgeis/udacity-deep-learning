FROM gcr.io/tensorflow/tensorflow:latest-py3
RUN pip install pandas
RUN pip install TFLearn h5py
WORKDIR /jnotebooks
