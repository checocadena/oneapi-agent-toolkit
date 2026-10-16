#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "adam_input.hpp"
#include "reference.h"
int main(int argc,char**argv){
  int N=1048576, T=10;
  std::vector<float> m(N),v(N),g(N),p(N);
  fill_input(m.data(),v.data(),g.data(),p.data(),N);
  const float step_size=1e-3f,decay=0.5f,b1=0.9f,b2=0.999f,eps=1e-8f,gs=256.f;
  for(size_t j=0;j<(size_t)N;j++){
    for(int t=1;t<=T;t++){
      float sg=g[j]/gs;
      m[j]=b1*m[j]+(1.f-b1)*sg;
      v[j]=b2*v[j]+(1.f-b2)*sg*sg;
      float mc=m[j]/(1.f-powf(b1,t));
      float vc=v[j]/(1.f-powf(b2,t));
      float denom=sqrtf(vc+eps);
      float upd=(mc/denom)+(decay*p[j]);
      p[j]-=(step_size*upd);
    }
  }
  FILE*f=fopen(argv[1],"wb"); fwrite(p.data(),4,N,f); fclose(f);
}
