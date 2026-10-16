#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "softmax_input.hpp"
int main(int argc,char**argv){
  int ns=atoi(argv[1]), sz=atoi(argv[2]);
  std::vector<float> in((size_t)ns*sz), ref((size_t)ns*sz), got((size_t)ns*sz);
  fill_input(in.data(), ns*sz);
  for(int i=0;i<ns;i++){
    double m=in[(size_t)i*sz]; for(int j=0;j<sz;j++) m=std::fmax(m,in[(size_t)i*sz+j]);
    double s=0; for(int j=0;j<sz;j++) s+=std::exp(in[(size_t)i*sz+j]-m);
    for(int j=0;j<sz;j++) ref[(size_t)i*sz+j]=std::exp(in[(size_t)i*sz+j]-m)/s;
  }
  FILE*f=fopen(argv[3],"rb"); fread(got.data(),4,got.size(),f); fclose(f);
  double md=0; for(size_t k=0;k<got.size();k++) md=std::fmax(md,std::fabs(got[k]-ref[k]));
  printf("%s max_abs_diff=%g\n", argv[3], md);
}
