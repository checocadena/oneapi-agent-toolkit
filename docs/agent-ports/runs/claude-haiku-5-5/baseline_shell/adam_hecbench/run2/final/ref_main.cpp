#include <cstdio>
#include <cstdlib>
#include <vector>
#include "adam_input.hpp"
#include "reference.h"
int main(int argc,char**argv){
  int n=atoi(argv[2]), ts=atoi(argv[3]);
  std::vector<float> m(n),v(n),g(n),p(n);
  fill_input(m.data(),v.data(),g.data(),p.data(),n);
  reference<float,float>(1,p.data(),m.data(),v.data(),g.data(),0.9f,0.999f,1e-8f,256.f,1e-3f,ts,n,ADAM_MODE_0,0.5f);
  FILE*f=fopen(argv[1],"wb"); fwrite(p.data(),4,n,f); fclose(f);
}
