{pkgs}: {
  deps = [
    pkgs.zlib
    pkgs.which
    pkgs.snappy
    pkgs.openssl
    pkgs.nsync
    pkgs.libjpeg_turbo
    pkgs.jsoncpp
    pkgs.grpc
    pkgs.gitFull
    pkgs.giflib
    pkgs.double-conversion
    pkgs.curl
    pkgs.bazel
  ];
}
