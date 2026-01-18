#!/bin/bash

docker build -t solar_energy .
docker run -v $(pwd)/results:/app/results solar_energy
