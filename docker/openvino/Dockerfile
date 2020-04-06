FROM ubuntu:18.04 AS openvino-base
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    libpython3.6 \
    sudo \
    python3-pip \
    python-dev \
    python-setuptools \
    libgtk2.0-dev \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

FROM openvino-base as openvino-inference
RUN echo "alias python=python3" >> /etc/bash.bashrc
RUN python3.6 -m pip install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install numpy
RUN pip3 install opencv-python==4.2.0.34
RUN pip3 install tensorflow==1.14.0
RUN pip3 install Keras==2.3.1
RUN pip3 install Pillow==7.1.1
RUN pip3 install scikit-learn==0.22.2
WORKDIR /app/

RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> /etc/bash.bashrc
ENV PYTHONPATH=$PYTHONPATH:/opt/intel/openvino/python/python3.6
ENV PORT 8080

EXPOSE 8080
COPY /src/ /app/src
COPY /config/ /app/config

