#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
int main(int argc,char**argv){
  int n = argc>2? atoi(argv[2]) : (1<<20);
  std::vector<float> x(n), y(n), out(n);
  for(int i=0;i<n;++i){ x[i]=static_cast<float>(i%1000)*0.001f; y[i]=static_cast<float>((i*7)%1000)*0.002f; }
  const float a=2.5f;
  for(int i=0;i<n;++i) out[i]=a*x[i]+y[i];
  FILE*f=fopen(argv[1],"rb"); std::vector<float> got(n);
  size_t r=fread(got.data(),sizeof(float),n,f); fclose(f);
  if(r!=(size_t)n){printf("short read\n");return 1;}
  double maxd=0; for(int i=0;i<n;++i) maxd=std::fmax(maxd,std::fabs(got[i]-out[i]));
  printf("max_abs_diff=%g\n",maxd);
  return maxd<=1e-6?0:1;
}
