#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <stdlib.h>

struct test1 {
  int x;
};

using test2_t = test1;

int main() {
  std::unique_ptr<test2_t> x(new test1());

  std::cout << "garbage memory: " << x->x << std::endl;

  std::vector<int> k;
  int sz = rand() % 5;
  for (int i = 0; i < sz; ++i) {
    k.push_back((x->x * i) << 1);
  }

  for (int i = 0; i < sz; ++i) {
    std::cout << "yes: " << k[i] << std::endl;
  }

  return 0;
}
