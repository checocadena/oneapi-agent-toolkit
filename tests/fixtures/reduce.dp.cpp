#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

inline dpct::constant_memory<float, 1> scale(1);

void reduce(const float* in, float* out, int n, float const *scale) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  float v = 0.f;
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
       i < n; i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
    v += in[i] * scale[0];
  for (int off = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_range()
                     .get(0) /
                 2;
       off > 0; off /= 2)
    v += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
  if ((item_ct1.get_local_id(2) &
       (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_local_range()
            .get(0) -
        1)) == 0)
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out,
                                                                         v);
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const int n = 1 << 20;
  float *d_in, *d_out;
  d_in = sycl::malloc_device<float>(n, q_ct1);
  d_out = sycl::malloc_device<float>(1, q_ct1);
  float s = 0.5f;
  q_ct1.memcpy(scale.get_ptr(), &s, sizeof(float)).wait();
  q_ct1.memset(d_out, 0, sizeof(float)).wait();
  dpct::event_ptr start, stop;
  start = new sycl::event(); stop = new sycl::event();
  dpct::sync_barrier(start);
  {
    scale.init();

    q_ct1.submit([&](sycl::handler &cgh) {
      auto scale_ptr_ct1 = scale.get_ptr();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1)
                           [[sycl::reqd_sub_group_size(32)]] {
                             reduce(d_in, d_out, n, scale_ptr_ct1);
                           });
    });
  }
  dpct::sync_barrier(stop);
  stop->wait_and_throw();
  float ms = 0; ms = (stop->get_profiling_info<
                          sycl::info::event_profiling::command_end>() -
                      start->get_profiling_info<
                          sycl::info::event_profiling::command_start>()) /
                     1000000.0f;
  /*
  DPCT1010:0: SYCL uses exceptions to report errors and does not use the error
  codes. The cudaGetLastError function call was replaced with 0. You need to
  rewrite this code.
  */
  dpct::err0 err = 0;
  /*
  DPCT1009:1: SYCL reports errors using exceptions and does not use error codes.
  Please replace the "get_error_string_dummy(...)" with a real error-handling
  function.
  */

  float h = 0; q_ct1.memcpy(&h, d_out, sizeof(float)).wait();
  int dev; dev = dpct::get_current_device_id(); dpct::device_info prop;
      dpct::get_device(dev).get_device_info(prop);
  printf("%s %f %f\n", prop.get_name(), h, ms);
  dpct::dpct_free(d_in, q_ct1); dpct::dpct_free(d_out, q_ct1);
  return 0;
}
