//===- GEMMRecognizer.cpp --- GEMM recognizer pass
//------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to recognize GEMM operations
//
//===----------------------------------------------------------------------===//

#include <string>
#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

using namespace mlir;
using namespace std;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

using llvm::dbgs;
#define DEBUG_TYPE "affine-gemm-recognizer"

namespace {

	struct GEMMRecognizer : public GEMMRecognizerBase<GEMMRecognizer> {
		GEMMRecognizer() = default;
		void runOnFunction() override;
	};

} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createGEMMRecognizerPass() {
	return std::make_unique<GEMMRecognizer>();
}

struct GEMMOperand {
	bool isGEMM;
	int64_t M, N, K;
	Value CMemRef;
	Value AMemRef;
	Value BMemRef;
	AffineLoadOp ALoadOp;
	AffineLoadOp BLoadOp;
	AffineStoreOp CStoreOp;
	AffineForOp MForOp, NForOp, KForOp;
};
typedef struct GEMMOperand GEMMOperand;

bool doArrayIndexesMatch(AffineForOp forOp1, AffineForOp forOp2,
	AffineLoadOp loadOp) {
	ValueRange mapOperands = loadOp.getMapOperands();
	if (forOp1.getBody()->getArgument(0) == mapOperands[0] &&
		forOp2.getBody()->getArgument(0) == mapOperands[1]) {
		return true;
	}

	return false;
}

int64_t getConstantRangeOfForLoop(AffineForOp forOp) {

	if (forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound()) {
		return (forOp.getConstantUpperBound() -
			forOp.getConstantLowerBound());
	}

	AffineMap lb = forOp.getLowerBoundMap();
	AffineMap ub = forOp.getUpperBoundMap();

	LLVM_DEBUG(lb.print(dbgs() << "\nThe lower bound:\n"));
	LLVM_DEBUG(ub.print(dbgs() << "\nThe upper bound:\n"));

	if (lb.getResults().size() == 1 && ub.getResults().size() == 1) {
		AffineExpr resultExpr = ub.getResults()[0] - lb.getResults()[0];
		LLVM_DEBUG(resultExpr.print(dbgs() << "\nThe result expression:\n"));

		// TODO: Check if the domains of the lb and ub maps are also the same.
		// Here we are assuming them to the same and that should be the case for 
		// an AffineForOp's lb and ub.

		if (resultExpr.isSymbolicOrConstant()) {
			LLVM_DEBUG(dbgs() << "The resultExpr isSymbolicOrConstant\n");
			if (resultExpr.getKind() == AffineExprKind::Constant) {
				int64_t range = resultExpr.cast<AffineConstantExpr>().getValue();
				string debugMessage = "\nThe range is: " + to_string(range) + "\n";
				LLVM_DEBUG(dbgs() << debugMessage.c_str());
				return range;
			}
		}
	}

	return -1;
}

GEMMOperand isAGEMMLoopNest(AffineForOp forOp1) {
	GEMMOperand gemmOperand;
	gemmOperand.isGEMM = false;

	Block &body1 = forOp1.region().front();
	if (auto forOp2 = dyn_cast<AffineForOp>(body1.front())) {
		Block &body2 = forOp2.region().front();
		if (auto forOp3 = dyn_cast<AffineForOp>(body2.front())) {
			LLVM_DEBUG(forOp1.getOperation()->print(
				dbgs() << "The triple nested loop is\n"));

			int64_t forOp1LoopRange = -1;
			int64_t forOp2LoopRange = -1;
			int64_t forOp3LoopRange = -1;

			if (forOp1.getStep() == 1) {
				forOp1LoopRange = getConstantRangeOfForLoop(forOp1);
				if (forOp1LoopRange != -1) {
					if (forOp2.getStep() == 1) {
						forOp2LoopRange = getConstantRangeOfForLoop(forOp2);
						if (forOp2LoopRange != -1) {
							if (forOp3.getStep() == 1) {
								forOp3LoopRange = getConstantRangeOfForLoop(forOp3);

								if (forOp3LoopRange != -1) {
									LLVM_DEBUG(dbgs() << "All 3 loops have the stride of 1.\n");
									LLVM_DEBUG(dbgs() << "forOp1LoopRange: " << forOp1LoopRange << "\n");
									LLVM_DEBUG(dbgs() << "forOp2LoopRange: " << forOp2LoopRange << "\n");
									LLVM_DEBUG(dbgs() << "forOp3LoopRange: " << forOp3LoopRange << "\n");

									// The last Op will be affine.terminator. Therefore, skipping that.
									LLVM_DEBUG(
										dbgs() << " num_ops: "
										<< forOp3.getOperation()->getBlock()->getOperations().size()
										<< "\n");

									if (forOp3.getOperation()->getBlock()->getOperations().size() == 2) {
										Block &body3 = forOp3.region().front();
										AffineLoadOp loadOp1, loadOp2, loadOp3;
										AffineForOp MForOp, NForOp, KForOp;
										auto range = llvm::make_range(body3.getOperations().begin(),
											std::prev(body3.getOperations().end()));

										int numLoads = 0, numStores = 0, numAdds = 0, numMuls = 0,
											numOthers = 0;
										for (Operation &op : range) {
											LLVM_DEBUG(op.print(dbgs() << "\nOp:\n"));

											OperationName name = op.getName();
											StringRef nameString = name.getStringRef();
											LLVM_DEBUG(dbgs() << "\n Operation Name: " << nameString);

											if (nameString.contains(".load")) {
												AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(op);

												if (numLoads == 0) {
													loadOp1 = loadOp;
												}
												else if (numLoads == 1) {
													loadOp2 = loadOp;
												}
												else if (numLoads == 2) {
													loadOp3 = loadOp;
												}

												numLoads++;
											}
											else if (nameString.contains(".store")) {
												AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(op);
												LLVM_DEBUG(
													storeOp.getMemRef().print(dbgs()
														<< "\ngetMemRef():\n"));
												LLVM_DEBUG(dbgs() << "MemRef\n: "
													<< storeOp.getMemRef());
												Value memRef = storeOp.getMemRef();
												gemmOperand.CMemRef = memRef;
												gemmOperand.CStoreOp = storeOp;
												numStores++;
												ValueRange mapOperands = storeOp.getMapOperands();

												LLVM_DEBUG(dbgs()
													<< "The map operands are: \n");
												for (Value val : mapOperands) {
													LLVM_DEBUG(val.print(dbgs() << "\nValue:\n"));
												}

												// Check if the array index variable is the same as the loop
												// variable C[i][j] : C[M][N]
												bool forOp1Taken = false, forOp2Taken = false,
													forOp3Taken = false;

												if (forOp1.getBody()->getArgument(0) ==
													mapOperands[0]) {
													gemmOperand.M = forOp1LoopRange;  																		forOp1Taken = true;
													MForOp = forOp1;
													LLVM_DEBUG(dbgs()
														<< "forOp1 loop variable = store's first index \n");
												}
												else if (forOp2.getBody()->getArgument(0) == mapOperands[0]) {
													gemmOperand.M = forOp2LoopRange;
													forOp2Taken = true;
													MForOp = forOp2;
													LLVM_DEBUG(dbgs()
														<< "forOp2 loop variable = store's first index \n");
												}
												else if (forOp3.getBody()->getArgument(0)
													== mapOperands[0]) {
													gemmOperand.M = forOp3LoopRange;
													forOp3Taken = true;
													MForOp = forOp3;
													LLVM_DEBUG(dbgs()
														<< "forOp3 loop variable = store's first index \n");
												}

												// Check if the array index variable is the same as the loop
												// variable
												if (forOp1.getBody()->getArgument(0) == mapOperands[1]) {
													gemmOperand.N = forOp1LoopRange;
													forOp1Taken = true;
													NForOp = forOp1;
													LLVM_DEBUG(dbgs()
														<< "forOp1 loop variable = store's second index \n");
												}
												else if (forOp2.getBody()->getArgument(0) == mapOperands[1]) {
													gemmOperand.N = forOp2LoopRange; 																		forOp2Taken = true;
													NForOp = forOp2;
													LLVM_DEBUG(dbgs()
														<< "forOp2 loop variable = store's second index \n");
												}
												else if (forOp3.getBody()->getArgument(0) == mapOperands[1]) {
													gemmOperand.N = forOp3LoopRange;
													forOp3Taken = true;
													NForOp = forOp3;
													LLVM_DEBUG(dbgs()
														<< "forOp3 loop variable = store's second index \n");
												}

												if (forOp1Taken == false) {
													gemmOperand.K = forOp1LoopRange;
													KForOp = forOp1;
												}

												if (forOp2Taken == false) {
													gemmOperand.K = forOp2LoopRange;
													KForOp = forOp2;
												}

												if (forOp3Taken == false) {
													gemmOperand.K = forOp3LoopRange;
													KForOp = forOp3;
												}

											}
											else if (nameString.contains(".add")) {
												numAdds++;
											}
											else if (nameString.contains(".mul")) {
												numMuls++;
											}
											else {
												numOthers++;
											}
										}

										if (numLoads == 3 && numStores == 1 && numAdds == 1 && numMuls == 1 &&
											numOthers == 0) {
											// Matrix multiplication pattern has been found.
											gemmOperand.isGEMM = true;

											LLVM_DEBUG(dbgs() << "K: gemmOperand.K: " << gemmOperand.K);

											// C[M][N] += A[M][K] * B[K][N];
											// We need to figure out which of the three loads is A and which one
											// is B.

											// Let's find A[M][K]
											if (doArrayIndexesMatch(MForOp, KForOp, loadOp1)) {
												gemmOperand.AMemRef = loadOp1.getMemRef();
												gemmOperand.ALoadOp = loadOp1;

											}
											else if (doArrayIndexesMatch(MForOp, KForOp, loadOp2)) {
												gemmOperand.AMemRef = loadOp2.getMemRef();
												gemmOperand.ALoadOp = loadOp2;
											}
											else if (doArrayIndexesMatch(MForOp, KForOp, loadOp3)) {
												gemmOperand.AMemRef = loadOp3.getMemRef();
												gemmOperand.ALoadOp = loadOp3;
											}

											// Let's find B[K][N]
											if (doArrayIndexesMatch(KForOp, NForOp, loadOp1)) {
												gemmOperand.BMemRef = loadOp1.getMemRef();
												gemmOperand.BLoadOp = loadOp1;
											}
											else if (doArrayIndexesMatch(KForOp, NForOp, loadOp2)) {
												gemmOperand.BMemRef = loadOp2.getMemRef();
												gemmOperand.BLoadOp = loadOp2;
											}
											else if (doArrayIndexesMatch(KForOp, NForOp, loadOp3)) {
												gemmOperand.BMemRef = loadOp3.getMemRef();
												gemmOperand.BLoadOp = loadOp3;
											}

											gemmOperand.MForOp = MForOp;
											gemmOperand.NForOp = NForOp;
											gemmOperand.KForOp = KForOp;
											return gemmOperand;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return gemmOperand;
}

void createSubViewOp(OpBuilder &bIn, Location locIn, Value memRef,
	int64_t size1, int64_t size2, AffineLoadOp loadOp,
	AffineForOp forOp1, AffineForOp forOp2) {
	ScopedContext scope(bIn, locIn);
	auto &b = ScopedContext::getBuilderRef();
	auto loc = ScopedContext::getLocation();

	auto arrayIndex1 = b.create<AffineApplyOp>(loc,
		forOp1.getLowerBoundMap(),
		forOp1.getLowerBoundOperands());
	auto arrayIndex2 = b.create<AffineApplyOp>(loc,
		forOp2.getLowerBoundMap(),
		forOp2.getLowerBoundOperands());
	LLVM_DEBUG(dbgs() << "arrayIndex1: " << arrayIndex1);
	LLVM_DEBUG(dbgs() << "arrayIndex2: " << arrayIndex2);

	SmallVector<Value, 4> offsets, sizes, strides;

	sizes.push_back(std_constant_index(size1));
	sizes.push_back(std_constant_index(size2));

	offsets.push_back(arrayIndex1);
	offsets.push_back(arrayIndex2);

	strides.push_back(std_constant_index(1));
	strides.push_back(std_constant_index(1));

	auto subView = b.create<SubViewOp>(loc, memRef, offsets, sizes, strides);
	auto elementType = b.getF32Type();
	auto unrankedType = UnrankedMemRefType::get(elementType, /*memorySpace*/ 0);
	auto unRankedMemRef = b.create<MemRefCastOp>(loc, subView, unrankedType);

}

void GEMMRecognizer::runOnFunction() {
	LLVM_DEBUG(dbgs() << "Running the GEMM recognizer pass \n");

	FuncOp f = getFunction();

	f.walk([&](AffineForOp forOp) {
		GEMMOperand gemmOperand = isAGEMMLoopNest(forOp);
		if (gemmOperand.isGEMM) {
			LLVM_DEBUG(dbgs() << "GEMM pattern has been FOUND\n");
			LLVM_DEBUG(dbgs() << "gemmOperand.M: " << gemmOperand.M << "\n");
			LLVM_DEBUG(dbgs() << "gemmOperand.N: " << gemmOperand.N << "\n");
			LLVM_DEBUG(dbgs() << "gemmOperand.K: " << gemmOperand.K << "\n");
			// Now we want to call a matrix multiplication routine here.
			OpBuilder b(forOp);

			/*
			SmallVector<Value, 6> ops;
			ops.push_back(gemmOperand.CMemRef);
			ops.push_back(gemmOperand.CMemRef);
			ops.push_back(gemmOperand.CMemRef);
			*/

			//TODO: Get the actual data type of the tensors gemmOperand.CMemRef
			auto elementType = b.getF32Type();
			auto unrankedType = UnrankedMemRefType::get(elementType, /*memorySpace*/ 0);

			createSubViewOp(b, forOp.getLoc(),
				gemmOperand.AMemRef, gemmOperand.M, gemmOperand.K, gemmOperand.ALoadOp,
				gemmOperand.MForOp, gemmOperand.KForOp);

			auto AMemRef = b.create<MemRefCastOp>(forOp.getLoc(),
				gemmOperand.AMemRef,
				unrankedType);

			auto BMemRef = b.create<MemRefCastOp>(forOp.getLoc(),
				gemmOperand.BMemRef,
				unrankedType);

			auto CMemRef = b.create<MemRefCastOp>(forOp.getLoc(),
				gemmOperand.CMemRef,
				unrankedType);

			auto op = b.create<PolyDLGEMMOp>(
				forOp.getLoc(), AMemRef, BMemRef,
				CMemRef, gemmOperand.M, gemmOperand.N, gemmOperand.K);

			LLVM_DEBUG(dbgs() << "CallOp: " << op);
			forOp.erase();
		}
		else {
			LLVM_DEBUG(dbgs() << "NOT a GEMM pattern \n");
		}
	});
}
