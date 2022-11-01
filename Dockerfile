#FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
WORKDIR ./scoped_persistency
RUN apt-get update \
  && apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev git \
  && apt-get install -y python3 python3-matplotlib python3-numpy
COPY . .
