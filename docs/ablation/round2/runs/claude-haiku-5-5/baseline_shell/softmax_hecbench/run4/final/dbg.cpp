#include <sycl/sycl.hpp>
#include <cstdio>
int main(){ sycl::queue q; printf("sg sizes:"); for(auto s: q.get_device().get_info<sycl::info::device::sub_group_sizes>()) printf(" %zu", s); printf("\n");
 size_t *o = sycl::malloc_shared<size_t>(4, q);
 q.parallel_for(sycl::nd_range<1>(256,256), [=](sycl::nd_item<1> it){ auto sg=it.get_sub_group(); if(it.get_global_id(0)==0){ o[0]=sg.get_local_linear_range(); o[1]=sg.get_group_range().size(); o[2]=sg.get_group_linear_range(); o[3]=sg.get_max_local_range()[0];} }).wait();
 printf("local=%zu groups_in_wg=%zu grouplin=%zu max=%zu\n",o[0],o[1],o[2],o[3]); }
