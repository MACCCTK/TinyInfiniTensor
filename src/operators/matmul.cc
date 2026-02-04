#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // =================================== 作业
  // ===================================
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  // =================================== 作业
  const Shape &shapeA = inputs[0]->getDims();
  const Shape &shapeB = inputs[1]->getDims();

  size_t rankA = shapeA.size();
  size_t rankB = shapeB.size();

  IT_ASSERT(rankA >= 2 && rankB >= 2, "MatMul requires at least 2D tensors");

  int dimA_M = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
  int dimA_K = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
  int dimB_K = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
  int dimB_N = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];

  IT_ASSERT(dimA_K == dimB_K,
            "MatMul inner dimensions should match: " + std::to_string(dimA_K) +
                " - " + std::to_string(dimB_K));

  Shape iterA(shapeA.begin(), shapeA.end() - 2);
  Shape iterB(shapeB.begin(), shapeB.end() - 2);

  Shape interim;
  if (iterA.empty() && iterB.empty()) {
    interim = {};
  } else if (iterA.empty()) {
    interim = iterB;
  } else if (iterB.empty()) {
    interim = iterA;
  } else {
    interim = infer_broadcast(iterA, iterB);
  }

  Shape output = interim;
  output.push_back(dimA_M);
  output.push_back(dimB_N);

  return {{output}};
}

} // namespace infini