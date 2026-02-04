#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来分配内存，返回起始地址偏移量
  // =================================== 作业
  // ===================================

  IT_ASSERT(size > 0, "Cannot allocate zero bytes");

  for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
    size_t blockAddr = it->first;
    size_t blockSize = it->second;

    if (blockSize >= size) {
      freeBlocks.erase(it);

      if (blockSize > size) {
        // split the block
        freeBlocks[blockAddr + size] = blockSize - size;
      }
      used += size;

      return blockAddr;
    }
  }

  size_t addr = peak;
  peak += size;
  used += size;

  return addr;
}

void Allocator::free(size_t addr, size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来回收内存
  // =================================== 作业
  // ===================================

  IT_ASSERT(size > 0, "Cannot free zero bytes");
  IT_ASSERT(addr + size <= peak, "free range out of peak");
  IT_ASSERT(used >= size, "double free or wrong size");

  used -= size;

  if (addr + size == peak) {
    peak = addr;

    while (!freeBlocks.empty()) {
      auto it = freeBlocks.end();
      --it;
      if (it->first + it->second == peak) {
        peak = it->first;
        freeBlocks.erase(it);
      } else {
        break;
      }
    }

    return;
  }

  // first freeblock created after first free
  freeBlocks[addr] = size;

  auto it = freeBlocks.find(addr);

  auto nextIt = std::next(it);
  if (nextIt != freeBlocks.end() && addr + size == nextIt->first) {
    size += nextIt->second;
    freeBlocks.erase(nextIt);
    freeBlocks[addr] = size;
  }

  if (it != freeBlocks.begin()) {
    auto prevIt = std::prev(it);
    if (prevIt->first + prevIt->second == addr) {
      size_t newAddr = prevIt->first;
      size_t newSize = prevIt->second + size;
      freeBlocks.erase(prevIt);
      freeBlocks.erase(it);
      freeBlocks[newAddr] = newSize;
    }
  }
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
