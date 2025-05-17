#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    for (auto [aElem, bElem] : llvm::zip(a, b)) {
      // Handle mixed precision: convert f16 inputs to f32 if needed
      Value convertedA = aElem;
      Value convertedB = bElem;
      if (accum.getType().isF32()) {
        if (aElem.getType().isF16()) {
          convertedA = builder.create<LLVM::FPExtOp>(loc, builder.getF32Type(), aElem);
        }
        if (bElem.getType().isF16()) {
          convertedB = builder.create<LLVM::FPExtOp>(loc, builder.getF32Type(), bElem);
        }
      }
      accum = builder.create<LLVM::FMulAddOp>(loc, convertedA, convertedB, accum);
    }
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
