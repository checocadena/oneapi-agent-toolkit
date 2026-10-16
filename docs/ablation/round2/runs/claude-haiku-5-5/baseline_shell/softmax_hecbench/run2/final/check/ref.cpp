// Reference: CPU softmax with expf, same math as the CUDA kernels. Compares candidate output files.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "softmax_input.hpp"

int main(int argc, char** argv) {
  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  const char* candFile = argv[3];
  int numElem = numSlice * sliceSize;
  std::vector<float> in(numElem), ref(numElem), cand(numElem);
  fill_input(in.data(), numElem);
  for (int i = 0; i < numSlice; i++) {
    const float* s = &in[i * sliceSize];
    float m = s[0];
    for (int j = 0; j < sliceSize; j++) m = std::fmax(m, s[j]);
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) sum += expf(s[j] - m);
    for (int j = 0; j < sliceSize; j++) ref[i * sliceSize + j] = expf(s[j] - m) / sum;
  }
  FILE* f = fopen(candFile, "rb");
  if (!f || fread(cand.data(), sizeof(float), numElem, f) != size_t(numElem)) {
    printf("read fail\n");
    return 1;
  }
  fclose(f);
  double maxErr = 0;
  for (int k = 0; k < numElem; k++) maxErr = std::fmax(maxErr, std::fabs(double(ref[k]) - cand[k]));
  printf("max_abs_err=%g %s\n", maxErr, maxErr <= 1e-6 ? "PASS" : "FAIL");
  return maxErr <= 1e-6 ? 0 : 2;
}
