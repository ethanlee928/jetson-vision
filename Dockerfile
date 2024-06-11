FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

ENV TZ=Asia/Hong_Kong
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo ${TZ} > /etc/timezone


# Install development dependencies
RUN apt update && apt install -y software-properties-common git cmake build-essential mesa-utils lsb-release

RUN add-apt-repository --remove "deb https://apt.kitware.com/ubuntu/ $(lsb_release --codename --short) main" && \
    apt install -y --no-install-recommends \
		  gstreamer1.0-tools \
		  gstreamer1.0-libav \
		  gstreamer1.0-rtsp \
		  gstreamer1.0-plugins-good \
		  gstreamer1.0-plugins-bad \
		  gstreamer1.0-plugins-ugly \
		  libgstreamer-plugins-base1.0-dev \
		  libgstreamer-plugins-good1.0-dev \
		  libgstreamer-plugins-bad1.0-dev && \
    if [ `lsb_release --codename --short` != 'bionic' ]; then \
    apt install -y --no-install-recommends \
		  gstreamer1.0-plugins-rtp; \
    else echo "skipping packages unavailable for Ubuntu 18.04"; fi

# Python Development Packages
RUN apt install -y libpython3-dev python3-numpy python3.8 python3.8-venv python3-venv python3.8-dev
ARG ENVDIR=/opt/sebit/venv
RUN mkdir -p ${ENVDIR} && python3.8 -m venv ${ENVDIR}
ENV PATH=${ENVDIR}/bin:$PATH
RUN chmod -R 777 ${ENVDIR}/
RUN pip install --upgrade pip && pip install Cython numpy opencv-python

# Build Jetson-inference from Source
RUN git clone --recursive --depth 1 https://github.com/dusty-nv/jetson-inference
RUN sed -i 's/nvcaffe_parser/nvparsers/g' /jetson-inference/CMakeLists.txt && \
	sed -i 's/PYTHON_BINDING_VERSIONS 2.7 3.6 3.7/PYTHON_BINDING_VERSIONS 3.8/g' /jetson-inference/python/CMakeLists.txt && \
	sed -i 's/PYTHON_BINDING_VERSIONS 2.7 3.6 3.7/PYTHON_BINDING_VERSIONS 3.8/g' /jetson-inference/utils/python/CMakeLists.txt && \
    sed -i 's|PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages|PYTHON_BINDING_INSTALL_DIR /opt/sebit/venv/lib/python3.8/site-packages|g' /jetson-inference/python/bindings/CMakeLists.txt && \
    sed -i 's|PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages|PYTHON_BINDING_INSTALL_DIR /opt/sebit/venv/lib/python3.8/site-packages|g' /jetson-inference/utils/python/bindings/CMakeLists.txt

RUN mkdir /jetson-inference/build
WORKDIR /jetson-inference/build
RUN cmake -DENABLE_NVMM=off ../
RUN make -j$(nproc)
RUN make install
RUN ldconfig

RUN pip install streamlit streamlit-webrtc supervision
## jetson-stats version should be identical to the host jetson-stats version
RUN pip install -U jetson-stats==4.2.2
WORKDIR /jetson-inference
