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
    python3-dev \
    python-setuptools \
    libgtk2.0-dev \
    gcc \
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
WORKDIR /app/
COPY /docker/openvino/requirements.txt /app/requirements.txt
RUN pip3 install -U setuptools
RUN pip3 install -r requirements.txt
RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> /etc/bash.bashrc

RUN mkdir /app/data
ENV PYTHONPATH=$PYTHONPATH:/opt/intel/openvino/python/python3.6
ENV LD_LIBRARY_PATH='/opt/intel/openvino_2020.1.023/opencv/lib:/opt/intel/openvino_2020.1.023/deployment_tools/ngraph/lib:/opt/intel/opencl: \
    /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/hddl/lib: \
    /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/gna/lib: \
    /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/mkltiny_lnx/lib: \
    /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/tbb/lib: \
    /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/lib/intel64:'
ENV APP_PORT 8080
ENV APP_HOST 0.0.0.0
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8080

COPY /model /app/model/
COPY /src/ /app/src
COPY /config/ /app/config
CMD gunicorn --bind :$APP_PORT src.inference_openvino:app --workers=4 --threads=4
