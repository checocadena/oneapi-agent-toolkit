#include <cmath>
#include <cstdio>
#include <vector>
int main(int c,char**v){
  FILE*a=fopen(v[1],"rb"),*b=fopen(v[2],"rb");
  std::vector<float> x(1<<26), y(1<<26);
  size_t na=fread(x.data(),4,x.size(),a), nb=fread(y.data(),4,y.size(),b);
  if(na!=nb){printf("size mismatch %zu %zu\n",na,nb);return 1;}
  double mx=0; for(size_t i=0;i<na;i++) mx=std::fmax(mx,std::fabs(x[i]-y[i]));
  printf("n=%zu max_abs_diff=%g %s\n",na,mx, mx<=1e-6?"PASS":"FAIL");
  return mx<=1e-6?0:1;
}
