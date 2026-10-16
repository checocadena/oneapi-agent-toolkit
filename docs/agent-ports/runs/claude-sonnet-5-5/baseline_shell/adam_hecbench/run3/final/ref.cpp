#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "adam_input.hpp"
#include "reference.h"
int main(int argc,char**argv){size_t n=1048576;std::vector<float>m(n),v(n),g(n),p(n);fill_input(m.data(),v.data(),g.data(),p.data(),n);
reference<float,float>(1,p.data(),m.data(),v.data(),g.data(),0.9f,0.999f,1e-8f,256.f,1e-3f,10,n,ADAM_MODE_0,0.5f);
FILE*f=fopen(argv[1],"wb");fwrite(p.data(),4,n,f);fclose(f);
FILE*a=fopen(argv[2],"rb");std::vector<float>c(n);fread(c.data(),4,n,a);double mx=0;size_t bad=0;for(size_t i=0;i<n;i++){double d=fabs(c[i]-p[i]);if(d>mx)mx=d;if(d>1e-5+1e-4*fabs(p[i]))bad++;}printf("maxdiff %g bad %zu\n",mx,bad);}
