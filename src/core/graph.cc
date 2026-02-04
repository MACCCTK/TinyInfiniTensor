#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"

#include <algorithm>
#include <numeric>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      if (auto const &inputs = op->getInputs();
          flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}

void GraphObj::optimize() {
  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose
  // 算子，且做的是相反的操作，可以将其全部删除）
  // 2.
  // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业
  // ===================================

  // Rule 1: Remove redundant transpose pairs
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = ops.begin(); it != ops.end(); ++it) {
      auto op = *it;
      if (op->getOpType() == OpType::Transpose) {
        auto transpose1 = as<TransposeObj>(op);
        auto output1 = transpose1->getOutput();
        auto targets = output1->getTargets();

        // Check if output has exactly one target and it's also a Transpose
        if (targets.size() == 1 &&
            targets[0]->getOpType() == OpType::Transpose) {
          auto transpose2 = as<TransposeObj>(targets[0]);
          auto perm1 = transpose1->getPermute();
          auto perm2 = transpose2->getPermute();

          bool shouldInverse = true;
          std::vector<int> tmp(perm1.size());
          for (size_t i = 0; i < perm1.size(); ++i) {
            tmp[i] = perm2[perm1[i]];
          }
          for (size_t i = 0; i < tmp.size(); ++i) {
            if (tmp[i] != (int)i) {
              shouldInverse = false;
              break;
            }
          }

          if (shouldInverse) {
            auto input1 = transpose1->getInputs(0);
            auto output2 = transpose2->getOutput();

            auto output2Targets = output2->getTargets();

            for (auto &t : output2Targets) {
              t->removePredecessors(transpose2);
              if (auto s = input1->getSource()) {
                t->addPredecessors(s);
                s->removeSuccessors(transpose1);
                s->addSuccessors(t);
              }

              for (auto &inp : t->inputs) {
                if (inp == output2) {
                  inp = input1;
                }
              }
              input1->addTarget(t);
            }

            output1->setSource(nullptr);
            output2->setSource(nullptr);

            input1->removeTarget(transpose1);

            removeTensor(output1);
            removeTensor(output2);
            removeOperator(transpose1);
            removeOperator(transpose2);

            changed = true;
            break;
          }
        }
      }
    }

    // help function to check if a permutation swaps last two dimensions
    auto shouldSwap = [](const std::vector<int> &perm) -> bool {
      if (perm.size() < 2)
        return false;
      for (size_t i = 0; i < perm.size() - 2; ++i) {
        if (perm[i] != (int)i)
          return false;
      }
      return perm[perm.size() - 2] == (int)(perm.size() - 1) &&
             perm[perm.size() - 1] == (int)(perm.size() - 2);
    };

    // help function to merge transpose into matmul
    auto mergeTransposeIntoMatmul = [&](const Ref<MatmulObj> &matmul,
                                        int inputIdx, bool flag) -> bool {
      auto input = matmul->getInputs(inputIdx);
      auto s = input->getSource();
      if (!s || s->getOpType() != OpType::Transpose)
        return false;

      auto transpose = as<TransposeObj>(s);
      if (!shouldSwap(transpose->getPermute()))
        return false;

      auto transposeInput = transpose->getInputs(0);
      matmul->inputs[inputIdx] = transposeInput;
      if (flag)
        matmul->setTransA(true);
      else
        matmul->setTransB(true);

      matmul->removePredecessors(transpose);
      if (auto s = transposeInput->getSource()) {
        matmul->addPredecessors(s);
        s->removeSuccessors(transpose);
        s->addSuccessors(matmul);
      }

      transposeInput->removeTarget(transpose);
      transposeInput->addTarget(matmul);
      input->removeTarget(matmul);
      input->setSource(nullptr);

      if (input->getTargets().empty()) {
        removeTensor(input);
      }
      removeOperator(transpose);
      return true;
    };

    // Rule 2: Merge transpose into matmul's transA/transB
    changed = true;
    while (changed) {
      changed = false;
      for (auto it = ops.begin(); it != ops.end(); ++it) {
        auto op = *it;
        if (op->getOpType() == OpType::MatMul) {
          auto matmul = as<MatmulObj>(op);
          if (mergeTransposeIntoMatmul(matmul, 0, true) ||
              mergeTransposeIntoMatmul(matmul, 1, false)) {
            changed = true;
            break;
          }
        }
      }
    }
  }
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // =================================== 作业
  // ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给
  // tensor 绑定内存
  // =================================== 作业
  // ===================================

  std::map<Tensor, size_t> tensorOffsets;
  for (auto &tensor : tensors) {
    size_t offset = allocator.alloc(tensor->getBytes());
    tensorOffsets[tensor] = offset;
  }

  void *basePtr = allocator.getPtr();

  for (auto &tensor : tensors) {
    size_t offset = tensorOffsets[tensor];
    tensor->setDataBlob(
        make_ref<BlobObj>(runtime, static_cast<uint8_t *>(basePtr) + offset));
  }

  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini