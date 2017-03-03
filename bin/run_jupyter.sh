#!/usr/bin/env bash

jupyter notebook "$@" --certfile=/jupyter.pem --keyfile /jupyter.key --port=443 --ip=0.0.0.0

