# syntax=docker/dockerfile:1
# villa 47111d1a VC3D apps for gpchain's render (vc_render_tifxyz). This is the `vc3d` stage of the recipe that made
# the render binary of every stored gpchain run (monorepo root Dockerfile, image inkcheck:ves572, commit b54f82e;
# unchanged since), copied here as its own file: Debian 13 by digest, apt from the Debian snapshot 20261005T000000Z,
# villa fetched from GitHub at the commit and checked by hash, the two moving CMake deps fetched at fixed commits.
# No villa source change. Build: scripts/build_villa.sh (context = this folder).
ARG DEBIAN=debian:trixie-20261005-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f
ARG DEBIAN_SNAPSHOT=20261005T000000Z

FROM ${DEBIAN} AS vc3d
ARG DEBIAN_SNAPSHOT
COPY apt-install.sh /usr/local/bin/apt-install.sh
# villa's build_from_src_debian.sh list, minus what only flatboi (PaStiX) needs.
RUN DEBIAN_SNAPSHOT="$DEBIAN_SNAPSHOT" sh /usr/local/bin/apt-install.sh \
      ca-certificates git build-essential cmake ninja-build pkg-config file \
      libavahi-client-dev libblosc-dev libboost-program-options-dev libboost-system-dev \
      libcgal-dev libceres-dev libcurl4-openssl-dev libgmp-dev liblapack-dev liblz4-dev \
      libmpfr-dev libopenblas-dev libopencv-contrib-dev libopencv-dev libsqlite3-dev \
      libsuitesparse-dev libtiff-dev libzstd-dev nlohmann-json3-dev qt6-base-dev zlib1g-dev

# Keep in step with VILLA_COMMIT in setup.sh. Sparse: volume-cartographer/ only (31 MiB).
ARG VILLA_COMMIT=47111d1a31381a5ff2c05046ed3e311b0dbeec64
# VC3D's CMake fetches two deps by moving refs (libbacktrace `master`, vc-delta3d tag v0.1.0).
# Both are fetched here at a fixed commit instead, so a rebuild cannot pick up new code.
ARG LIBBACKTRACE_COMMIT=0b9b49cf4a2c9229fc052d6716e1528b2f23e91a
ARG VC_DELTA3D_COMMIT=eedd0e62d988bcf97fbaee4ebb7af026f52ca49a
WORKDIR /opt/vc3d
RUN fetch() { git init -q "$1" && git -C "$1" remote add origin "$2" \
      && git -C "$1" fetch -q --depth 1 origin "$3" && git -C "$1" checkout -q FETCH_HEAD \
      && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && git init -q villa && git -C villa sparse-checkout set --no-cone /volume-cartographer/ \
 && fetch villa https://github.com/ScrollPrize/villa.git "$VILLA_COMMIT" \
 && mv villa/volume-cartographer src && rm -rf villa \
 && fetch deps/vc-delta3d https://github.com/ScrollPrize/vc-delta3d.git "$VC_DELTA3D_COMMIT" \
 && fetch deps/libbacktrace https://github.com/ianlancetaylor/libbacktrace.git "$LIBBACKTRACE_COMMIT"
# libbacktrace has no .pc file; with one, VC3D's CMake takes it as the system copy and skips its
# own fetch of `master`.
RUN cd deps/libbacktrace && ./configure -q --prefix=/opt/vc3d/deps/bt --enable-static \
      --disable-shared --with-pic && make -j 12 && make install \
 && mkdir -p /opt/vc3d/deps/bt/lib/pkgconfig \
 && printf '%s\n' 'prefix=/opt/vc3d/deps/bt' 'Name: libbacktrace' 'Description: libbacktrace' \
      "Version: ${LIBBACKTRACE_COMMIT}" 'Libs: -L${prefix}/lib -lbacktrace' \
      'Cflags: -I${prefix}/include' > /opt/vc3d/deps/bt/lib/pkgconfig/libbacktrace.pc

# The CLI apps to build. The two the researchers asked for first, then the tifxyz tools they
# use around them. Each extra one costs seconds once vc_core is built.
ARG VC3D_APPS="vc_grow_seg_from_seed vc_render_tifxyz vc_grow_seg_from_segments vc_tifxyz2obj \
vc_obj2tifxyz vc_calc_surface_metrics vc_tifxyz_winding vc_gen_normalgrids"
# VC3D's CMake keeps build-tree RPATHs (it forces CMAKE_BUILD_WITH_INSTALL_RPATH OFF), so the
# apps find /opt/vc3d/lib through ldconfig instead, here and in the final stage. The `grep` fails
# the build if CMake ever falls back to fetching libbacktrace. 12 build threads: the box is shared.
RUN PKG_CONFIG_PATH=/opt/vc3d/deps/bt/lib/pkgconfig cmake -S src -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBLA_VENDOR=OpenBLAS -DVC_BUILD_APPS=ON -DVC_BUILD_FLATBOI=OFF -DVC_BUILD_PYTHON=OFF \
      -DVC_TESTING=OFF -DVC_MARCH_NATIVE=OFF \
      -DVC_GIT_SHA1="$VILLA_COMMIT" \
      -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D=/opt/vc3d/deps/vc-delta3d \
      -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
    | tee /tmp/configure.log \
 && grep -q "Using system libbacktrace" /tmp/configure.log \
 && cmake --build build -j 12 --target $VC3D_APPS \
 && mkdir -p /opt/vc3d/bin /opt/vc3d/lib \
 && for a in $VC3D_APPS; do cp build/bin/"$a" /opt/vc3d/bin/; done \
 && find build -name 'lib*.so*' ! -path '*/CMakeFiles/*' -exec cp -a {} /opt/vc3d/lib/ ';' \
 && cp src/LICENSE src/NOTICE /opt/vc3d/ \
 && echo "$VILLA_COMMIT" > /opt/vc3d/COMMIT
# Debian packages the final image needs: every shared lib the apps load that is not one of
# VC3D's own, mapped to its package. Written to a file the final stage installs from; the build
# fails if a lib is missing or has no owning package.
RUN echo /opt/vc3d/lib > /etc/ld.so.conf.d/vc3d.conf && ldconfig \
 && ! ldd /opt/vc3d/bin/* | grep 'not found' \
 && for lib in $(ldd /opt/vc3d/bin/* /opt/vc3d/lib/*.so* | awk '/=> \// {print $3}' \
                 | grep -v '^/opt/vc3d/' | sort -u); do \
      real="$(readlink -f "$lib")"; \
      owner="$(dpkg -S "$real" 2>/dev/null || dpkg -S "${real#/usr}" 2>/dev/null \
               || dpkg -S "$lib" 2>/dev/null)" || { echo "no package owns $lib" >&2; exit 1; }; \
      echo "${owner%%:*}" >> /tmp/owners; \
    done \
 && sort -u /tmp/owners > /opt/vc3d/runtime-packages.txt \
 && echo "$(wc -l < /opt/vc3d/runtime-packages.txt) runtime packages:" \
 && cat /opt/vc3d/runtime-packages.txt
