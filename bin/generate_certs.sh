#!/usr/bin/env bash

openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout jupyter.key -out jupyter.pem
