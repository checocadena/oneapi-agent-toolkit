#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
void atomic_reduction(int *in, int *out, int arrayLength) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int sum = 0;
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  for (int i = idx; i < arrayLength;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
    sum+=in[i];
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out, sum);
}

void atomic_reduction_v2(int *in, int* out, int arrayLength) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int sum = 0;
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  for (int i = idx * 2; i < arrayLength;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2) * 2) {
    sum+=in[i] + in[i+1];
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out, sum);
}

void atomic_reduction_v4(int *in, int* out, int arrayLength) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int sum = 0;
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  for (int i = idx * 4; i < arrayLength;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2) * 4) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out, sum);
}

void atomic_reduction_v8(int *in, int* out, int arrayLength) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int sum = 0;
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  for (int i = idx * 8; i < arrayLength;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2) * 8) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7];
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out, sum);
}

void atomic_reduction_v16(int *in, int* out, int arrayLength) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int sum = 0;
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  for (int i = idx * 16; i < arrayLength;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2) * 16) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7]
      +in[i+8] +in[i+9] +in[i+10] +in[i+11] +in[i+12] +in[i+13] +in[i+14] +in[i+15] ;
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out, sum);
}

