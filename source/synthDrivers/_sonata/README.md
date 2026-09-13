# Sonata wire protocol

`sonata_grpc.proto` contains the wire definitions from
[Sonata revision 451f9ebf2bd2aa2ba1be25fcec3b7593eeabf6ee](https://github.com/mush42/sonata/blob/451f9ebf2bd2aa2ba1be25fcec3b7593eeabf6ee/crates/frontends/grpc/proto/sonata_grpc.proto).
It is distributed under the included `LICENSE.sonata` (MIT).

`sonata_grpc_pb2.py` is generated with `grpcio-tools==1.78.0` and is copied into
`extras/piperService/sonata_piper` so that the service package can be installed
without NVDA. The service package also includes the upstream license.
The client and service use generic gRPC calls with the original method names;
there is no custom wire protocol or generated gRPC stub dependency at runtime.

To regenerate from the repository root in an environment with `grpcio-tools==1.78.0`:

```powershell
python -m grpc_tools.protoc -I source/synthDrivers/_sonata --python_out=source/synthDrivers/_sonata source/synthDrivers/_sonata/sonata_grpc.proto
Copy-Item source/synthDrivers/_sonata/sonata_grpc_pb2.py extras/piperService/sonata_piper/sonata_grpc_pb2.py
```

Do not edit or reformat the generated module. Its runtime dependencies are
`grpcio` and `protobuf`; `grpcio-tools` is required only for regeneration.
