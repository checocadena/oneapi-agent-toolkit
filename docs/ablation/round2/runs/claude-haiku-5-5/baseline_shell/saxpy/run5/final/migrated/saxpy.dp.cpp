// CUDA SAXPY (the code to be ported). Unified memory, error checks, event timing.
// Usage: saxpy <out.bin> [n=1048576]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>

void saxpy(int n, float a, const float* x, float* y) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : (1 << 20);
  const float a = 2.5f;
  float *x, *y;
  x = sycl::malloc_shared<float>(n, q_ct1);
  y = sycl::malloc_shared<float>(n, q_ct1);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 1000) * 0.001f;
    y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
  }
  dpct::event_ptr start, stop;
  start = new sycl::event();
  stop = new sycl::event();
  dpct::sync_barrier(start);
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (n + 255) / 256) *
                                           sycl::range<3>(1, 1, 256),
                                       sycl::range<3>(1, 1, 256)),
                     [=](sycl::nd_item<3> item_ct1) {
                       saxpy(n, a, x, y);
                     });
  dpct::sync_barrier(stop);
  dpct::err0 err = DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw());

  float ms = 0.f;
  ms =
      (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
       start
           ->get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
  std::printf("kernel_ms=%f\n", ms);
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n)) return 1;
  std::fclose(f);
  dpct::dpct_free(x, q_ct1);
  dpct::dpct_free(y, q_ct1);
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
