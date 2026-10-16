#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "softmax_input.hpp"
int main(int c,char**v){int ns=atoi(v[2]),ss=atoi(v[3]);std::vector<float>x(ns*ss),g(ns*ss);fill_input(x.data(),ns*ss);
FILE*f=fopen(v[1],"rb");fread(g.data(),4,ns*ss,f);double m=0;
for(int i=0;i<ns;i++){double mx=-1e30,s=0;for(int j=0;j<ss;j++)mx=fmax(mx,x[i*ss+j]);for(int j=0;j<ss;j++)s+=exp(x[i*ss+j]-mx);
for(int j=0;j<ss;j++)m=fmax(m,fabs(exp(x[i*ss+j]-mx)/s-g[i*ss+j]));}
printf("maxerr %g\n",m);}
