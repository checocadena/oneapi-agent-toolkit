#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
int main(int argc,char**argv){
  int n = argc>2? std::atoi(argv[2]) : (1<<20);
  const float a=2.5f;
  std::vector<float> x(n),y(n);
  for(int i=0;i<n;++i){x[i]=static_cast<float>(i%1000)*0.001f; y[i]=static_cast<float>((i*7)%1000)*0.002f;}
  for(int i=0;i<n;++i) y[i]=a*x[i]+y[i];
  std::vector<float> got(n);
  FILE*f=std::fopen(argv[1],"rb"); size_t r=std::fread(got.data(),sizeof(float),n,f); std::fclose(f);
  if((int)r!=n){std::printf("short read %zu\n",r);return 1;}
  double maxd=0; for(int i=0;i<n;++i) maxd=std::fmax(maxd,std::fabs((double)got[i]-y[i]));
  std::printf("n=%d max_abs_diff=%g\n",n,maxd);
  return maxd<=1e-6?0:1;
}
