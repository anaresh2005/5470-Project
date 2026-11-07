// Check return codes without modifying David's generated source or interface.
#include "common.h"
namespace checked {
inline ncclResult_t group_start() { auto r = ncclGroupStart(); NCCL(r); return r; }
inline ncclResult_t group_end() { auto r = ncclGroupEnd(); NCCL(r); return r; }
inline ncclResult_t send(const void* b, size_t n, ncclDataType_t t, int peer,
                        ncclComm_t c, cudaStream_t s) {
  auto r = ncclSend(b, n, t, peer, c, s); NCCL(r); return r;
}
inline ncclResult_t recv(void* b, size_t n, ncclDataType_t t, int peer,
                        ncclComm_t c, cudaStream_t s) {
  auto r = ncclRecv(b, n, t, peer, c, s); NCCL(r); return r;
}
inline cudaError_t device(int d) { auto r = cudaSetDevice(d); CUDA(r); return r; }
}
#define ncclGroupStart checked::group_start
#define ncclGroupEnd checked::group_end
#define ncclSend checked::send
#define ncclRecv checked::recv
#define cudaSetDevice checked::device
#include "../gen/gen_n4.cu"
