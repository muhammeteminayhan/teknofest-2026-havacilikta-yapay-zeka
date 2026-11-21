# GOREV 2 — ORB-SLAM3 v1.0 (monokuler) cevrimdisi degerlendirme konteyneri.
# Ana Python ortamina dokunmaz. Kareler /veri altina baglanir.
#
#   docker build -t orbslam3:yerel gorev2/orbslam3_docker
#   docker run --rm -v <gorev2/outputs/final>:/veri orbslam3:yerel \
#       mono_nedensel /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /veri/colmap/<ad>/orbslam3.yaml \
#       /veri/colmap/<ad>/images /veri/colmap/<ad>/orbslam3_poz.csv 7.5
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libeigen3-dev libopencv-dev libglew-dev libboost-serialization-dev \
        libboost-filesystem-dev libssl-dev libepoxy-dev python3 \
    && rm -rf /var/lib/apt/lists/*

# Pangolin v0.6 (ORB-SLAM3 v1.0 ile Ubuntu 20.04'te sorunsuz derlenen surum)
RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin \
    && cd /opt/Pangolin && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    && make -j4 && make install && ldconfig

# ORB-SLAM3 v1.0. Ubuntu 20.04'te OpenCV 4.2 var; depo 4.4 istiyor -> 4.2'ye indir.
RUN git clone --depth 1 --branch v1.0-release https://github.com/UZ-SLAMLab/ORB_SLAM3.git /opt/ORB_SLAM3 \
    && cd /opt/ORB_SLAM3 \
    && sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/' CMakeLists.txt \
    && sed -i 's/-march=native//g' CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt \
    && sed -i 's/++11/++14/g' CMakeLists.txt

COPY mono_nedensel.cc /opt/ORB_SLAM3/Examples/Monocular/mono_nedensel.cc
RUN cd /opt/ORB_SLAM3 \
    && printf '\nadd_executable(mono_nedensel Examples/Monocular/mono_nedensel.cc)\ntarget_link_libraries(mono_nedensel ${PROJECT_NAME})\n' >> CMakeLists.txt \
    && cd Thirdparty/DBoW2 && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 \
    && cd /opt/ORB_SLAM3/Thirdparty/g2o && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 \
    && cd /opt/ORB_SLAM3/Thirdparty/Sophus && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF && make -j4 \
    && cd /opt/ORB_SLAM3/Vocabulary && tar -xf ORBvoc.txt.tar.gz \
    && cd /opt/ORB_SLAM3 && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 \
    && cp "$(find /opt/ORB_SLAM3 -name mono_nedensel -type f -perm -u+x | head -1)" /usr/local/bin/mono_nedensel

ENV LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib
WORKDIR /veri
