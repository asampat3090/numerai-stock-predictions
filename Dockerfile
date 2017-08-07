FROM cloudgear/ubuntu:14.04

# Install system requirements
RUN apt-get update; \
    apt-get install -y \
      python python-pip \
      build-essential \
      python-dev \
      python-setuptools \
      python-matplotlib \
      libatlas-dev \
      curl \
      libatlas3gf-base && \
    apt-get clean

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install basic python packages
RUN pip install numpy==1.13.1
RUN pip install scipy
RUN pip install -U scikit-learn
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install --pre xgboost

# Ensure proper installation
RUN update-alternatives --set libblas.so.3 \
      /usr/lib/atlas-base/atlas/libblas.so.3; \
    update-alternatives --set liblapack.so.3 \
     /usr/lib/atlas-base/atlas/liblapack.so.3
