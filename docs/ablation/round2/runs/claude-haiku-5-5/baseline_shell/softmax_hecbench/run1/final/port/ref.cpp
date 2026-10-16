#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "softmax_input.hpp"
int main(int argc,char**argv){
  int ns=atoi(argv[1]), sz=atoi(argv[2]);
  std::vector<float> in((size_t)ns*sz), out((size_t)ns*sz);
  fill_input(in.data(), ns*sz);
  for(int i=0;i<ns;i++){
    const float*s=&in[(size_t)i*sz]; float*d=&out[(size_t)i*sz];
    float m=s[0]; for(int j=0;j<sz;j++) m=std::max(m,s[j]);
    float sum=0; for(int j=0;j<sz;j++) sum+=expf(s[j]-m);
    for(int j=0;j<sz;j++) d[j]=expf(s[j]-m)/sum;
  }
  FILE*f=fopen(argv[3],"wb"); fwrite(out.data(),4,out.size(),f); fclose(f);
}
