# syntax=docker/dockerfile:1.7
# villa 6bbe6e21 VC3D tools for gpchain's isometric stage and band cuts (vc_tifxyz2obj, flatboi, vc_obj2tifxyz).
# The base and build stages are the recipe that made the binaries of every stored gpchain run (VES-1471, monorepo
# commit 92fba3b9, tools/docker/villa-ink-6bbe6e2/Dockerfile): Ubuntu 26.04 by digest, apt snapshot 20261022T000000Z,
# villa's own volume-cartographer/scripts/install_build_deps.sh (its AWS CLI block cut: unpinned and unused), the
# same cmake preset and target list. Only that recipe's Python / model stages are left out (gpchain does not use them).
# Build context: scripts/build_villa.sh (villa/volume-cartographer from `git archive 6bbe6e21`). No villa source change.
ARG UBUNTU=ubuntu:26.04@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944

FROM ${UBUNTU} AS base
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_SNAPSHOT=20261022T000000Z
COPY villa/volume-cartographer/scripts/install_build_deps.sh /tmp/install_build_deps.sh
RUN apt-get update -y && apt-get install -y --no-install-recommends ca-certificates \
 && echo "APT::Snapshot \"${APT_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50snapshot \
 && sed -i '/^# AWS CLI v2/,$d' /tmp/install_build_deps.sh \
 && bash /tmp/install_build_deps.sh \
 && rm -rf /var/lib/apt/lists/* /tmp/install_build_deps.sh

FROM base AS vcbuild
ARG JOBS=8
ARG VC_TARGETS="vc_render_tifxyz vc_tifxyz_trim vc_tifxyz2obj vc_obj2tifxyz vc_obj_uv_lift flatboi"
COPY villa/volume-cartographer /src
WORKDIR /src
RUN cmake --preset ci-release-gcc -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
 && cmake --build --preset ci-release-gcc -j "${JOBS}" --target ${VC_TARGETS} \
 && mkdir -p /out/bin /out/lib \
 && for t in ${VC_TARGETS}; do cp "build/ci-release-gcc/bin/$t" /out/bin/; done \
 && find build/ci-release-gcc -name 'lib*.so*' \( -type f -o -type l \) -exec cp -P {} /out/lib/ \; \
 && cp LICENSE /out/LICENSE \
 && ls -l /out/bin /out/lib

FROM base AS tools
COPY --from=vcbuild /out/bin/ /usr/local/bin/
COPY --from=vcbuild /out/lib/ /usr/local/lib/
COPY --from=vcbuild /out/LICENSE /usr/local/share/villa/LICENSE
COPY villa/COMMIT /usr/local/share/villa/COMMIT
RUN ldconfig && for b in /usr/local/bin/vc_* /usr/local/bin/flatboi; do \
      ! ldd "$b" | grep "not found" || exit 1; done
