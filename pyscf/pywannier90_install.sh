#!/bin/bash

# modify Makefile
cp ../Makefile ../Makefile.copy
cp Makefile ../Makefile

# modify make.inc
cp ../make.inc ../make.inc.copy
sed -i '/FCOPTS/d'  ../make.inc
sed -i '/LDOPTS/d'  ../make.inc
sed -i '/COMMS/d'  ../make.inc # do not use MPI
sed -i '/MPI/d'  ../make.inc
echo 'FCOPTS = -O2 -g -fPIC' >> ../make.inc 
echo 'LDOPTS = -shared' >> ../make.inc 

# modify Makefile.2
cp ../src/Makefile.2 ../src/Makefile.2.copy
cp Makefile.2 ../src/ 


cd ../
make clean
make libwannier.so -j 4

