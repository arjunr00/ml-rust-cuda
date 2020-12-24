current="$(pwd)"
cd "$(dirname $0)"/..
mkdir -p ./scripts/tmp
echo -n "Compiling src/prof.cu .. "
nvcc ./src/prof.cu -o ./scripts/tmp/prof
echo "done"
nvprof ./scripts/tmp/prof > ./scripts/tmp/prof.out
echo "Standard out written to scripts/tmp/prof.out"
if [ "$1" != "no-rm" ] ; then
  rm ./scripts/tmp/prof
else
  echo "[Not removing scripts/tmp/prof]"
fi
cd "$current"
