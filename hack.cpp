// Anaconda:     g++-7 -D_GLIBCXX_USE_CXX11_ABI=0 -Os -Wall -fPIC hack.cpp -c -o hack_abi_old.o
// Others:       g++-7 -D_GLIBCXX_USE_CXX11_ABI=1 -Os -Wall -fPIC hack.cpp -c -o hack_abi_new.o
// Linker:       g++-7 -fPIC -shared hack_abi_old.o hack_abi_new.o -o hack.so
// Strip:        strip hack.so

#include <atomic>
#include <string>
#include <random>
#include <stdint.h>
#include <unistd.h>

namespace at {

std::string NewProcessWideShmHandle()
{
  static std::atomic<uint64_t> counter{0};
  static std::random_device rd;
  std::string handle = "/torch_";
  handle += std::to_string(getpid());
  handle += "_";
  handle += std::to_string(rd());
  handle += "_";
  handle += std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
  return handle;
}

}
