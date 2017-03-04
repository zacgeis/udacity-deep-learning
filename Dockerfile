FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3
RUN pip install pandas
RUN pip install TFLearn h5py tqdm

COPY bin/run_jupyter.sh /
COPY bin/jupyter.key /
COPY bin/jupyter.pem /

WORKDIR /notebooks

EXPOSE 443

# CMD ["/bin/bash"]
CMD ["/run_jupyter.sh"]
