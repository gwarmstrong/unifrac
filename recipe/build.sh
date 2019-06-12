#!/bin/bash

set -x -e

echo "BUILDING..."
echo $CONDA_PREFIX
echo $CONDA_BUILD_SYSROOT

if [ "$(uname)" = "Linux" ] ; then
	SED='sed -i' ;
else 
	SED='sed -i '"'"''"'"' ' ;
fi

$SED 's/^CXXBASE=.*/CXXBASE=clang++/' `which h5c++`
$SED 's/^CXXLINKERBASE=.*/CXXLINKERBASE=clang++/' `which h5c++`

export CC=`which h5c++`


pushd sucpp; make test; make main; popd
$PYTHON -m pip install .

#pushd sucpp; make test; make main; make api; make rapi_test; popd # make capi_test; make rapi_test; popd 


#mkdir -p $PREFIX/bin
#ln -s ./sucpp/ssu $PREFIX/bin/ssu

which ssu
