// Multi-process NCCL send/recv test, with one MPI process per GPU.
// Every rank exchanges a buffer with every other rank, and each received value is checked.
// The rank count comes from MPI, so the same program works on one node or several.
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK_CUDA(cmd)                                                    \
  do {                                                                     \
    cudaError_t e = cmd;                                                   \
    if (e != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e),   \
              __FILE__, __LINE__);                                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                      \
  } while (0)

#define CHECK_NCCL(cmd)                                                    \
  do {                                                                     \
    ncclResult_t r = cmd;                                                  \
    if (r != ncclSuccess) {                                                \
      fprintf(stderr, "NCCL error: %s (%s:%d)\n", ncclGetErrorString(r),   \
              __FILE__, __LINE__);                                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                      \
  } while (0)

static const size_t MAX_COUNT = 10000000;
static const long MAX_PRINTED_ERRORS = 20;

// Encodes the sender, the round, and the position, so a misrouted or misplaced value is caught.
static int expected_value(int src, int shift, size_t i) {
  return (src + 1) * 100000000 + shift * 10000000 + (int)i;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t count = argc > 1 ? strtoull(argv[1], NULL, 10) : 1048576;
  if (count == 0 || count > MAX_COUNT || size < 2 || size > 8) {
    if (rank == 0) fprintf(stderr, "Need 2-8 ranks and 1-%zu ints per buffer\n", MAX_COUNT);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Each process on a node takes a different GPU.
  int nDevices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&nDevices));
  const char* localId = getenv("SLURM_LOCALID");
  int device = (localId ? atoi(localId) : 0) % nDevices;
  CHECK_CUDA(cudaSetDevice(device));

  char host[MPI_MAX_PROCESSOR_NAME];
  int hostLen;
  MPI_Get_processor_name(host, &hostLen);
  printf("Rank %d of %d on %s, GPU %d of %d\n", rank, size, host, device, nDevices);
  fflush(stdout);

  // Rank 0 creates the ID and MPI gives it to every other process.
  ncclUniqueId id;
  if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, size, id, rank));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  int* sendBuf;
  int* recvBuf;
  CHECK_CUDA(cudaMalloc(&sendBuf, count * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&recvBuf, count * sizeof(int)));
  std::vector<int> hostBuf(count);

  long errors = 0;
  for (int shift = 1; shift < size; ++shift) {
    int dst = (rank + shift) % size;
    int src = (rank - shift + size) % size;

    for (size_t i = 0; i < count; ++i) hostBuf[i] = expected_value(rank, shift, i);
    CHECK_CUDA(cudaMemcpy(sendBuf, hostBuf.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(recvBuf, 0, count * sizeof(int)));

    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclSend(sendBuf, count, ncclInt, dst, comm, stream));
    CHECK_NCCL(ncclRecv(recvBuf, count, ncclInt, src, comm, stream));
    CHECK_NCCL(ncclGroupEnd());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaMemcpy(hostBuf.data(), recvBuf, count * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      if (hostBuf[i] != expected_value(src, shift, i)) {
        if (errors < MAX_PRINTED_ERRORS) {
          printf("Rank %d, from %d, idx %zu, val %d\n", rank, src, i, hostBuf[i]);
        }
        ++errors;
      }
    }
  }

  long totalErrors = 0;
  MPI_Allreduce(&errors, &totalErrors, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    int exchanges = size * (size - 1);
    if (totalErrors == 0) {
      printf("PASS: %d ranks completed %d transfers of %zu ints with correct values\n",
             size, exchanges, count);
    } else {
      printf("FAIL: %ld wrong values\n", totalErrors);
    }
  }

  CHECK_CUDA(cudaFree(sendBuf));
  CHECK_CUDA(cudaFree(recvBuf));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_NCCL(ncclCommDestroy(comm));
  MPI_Finalize();
  return totalErrors == 0 ? 0 : 1;
}
