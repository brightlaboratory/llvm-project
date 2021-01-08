//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;

using llvm::dbgs;
#define DEBUG_TYPE "affine-polydl"

namespace {

	struct PolyDLPass : public PolyDLPassBase<PolyDLPass> {
		PolyDLPass() = default;
        void runOnFunction() override;
        void generateFuncCopies(FuncOp f, SmallVector<unsigned, 6> tileSizes, SmallVector<unsigned, 4> permMap);
        void statsWorkingSetSizes(std::vector<long int> bandFootprints);
        void computeWorkingSetSizes(ArrayRef<AffineForOp> band);
        void generate_ranking();
        void generateTop5pct(std::vector<SmallVector<unsigned, 6> > tileSizeOut,std::vector<SmallVector<unsigned, 4> > permMapOut);
        std::vector<std::vector<float>> dataset; 
	};

} // end anonymous namespace


std::unique_ptr<OperationPass<FuncOp>> mlir::createPolyDLPassPass() {
	return std::make_unique<PolyDLPass>();
}


void generatePermutationMaps(unsigned a[], int n,std::vector<SmallVector<unsigned, 4> > &permMapOut) 
{ 
    SmallVector<unsigned, 4> permMap;
  
    // Find all possible permutations  
    do { 

        for (int i = 0; i < n; i++)
            permMap.push_back(a[i]);

        permMapOut.push_back(permMap);

        for (int i = 0; i < n; i++)
            permMap.pop_back();

    } while (std::next_permutation(a, a + n)); 
} 

void generateTileSizes(int p, int n,std::vector<int> lowerBound,std::vector<int> upperBound, SmallVector<unsigned, 6> out,std::vector<SmallVector<unsigned, 6> > &tileSizeOut) 
{ 
      
    if (p>=n) 
    { 
        tileSizeOut.push_back(out);
        return; 
    }
    
    for (int i = lowerBound[p]; i <= upperBound[p]; i=i*2) 
    { 
        out.push_back(i);
        generateTileSizes(p+1,n,lowerBound,upperBound,out,tileSizeOut); 
        out.pop_back(); 
    } 
}

bool sortcol( const std::vector<float>& v1, 
               const std::vector<float>& v2 ) { 
 return v1[4] < v2[4]; 
} 

void PolyDLPass::statsWorkingSetSizes(std::vector<long int> bandFootprints) {
    
    long int L1CacheSize,L2CacheSize,L3CacheSize;
    // Stats generation of working set size
    if(explicit_cacheSize)
        dbgs()<<"explicit_cacheSize "<<explicit_cacheSize;
    
    if (!explicit_cacheSizes.empty() && explicit_cacheSizes.size()==3){
        SmallVector<unsigned, 4> Ex_cacheSize;        
        for(auto i : explicit_cacheSizes)
            Ex_cacheSize.push_back(i);
        // dbgs()<<"Cache Size "<< Ex_cacheSize[0] <<"\n" ;
        // dbgs()<<"Cache Size "<< Ex_cacheSize[1] <<"\n" ;
        // dbgs()<<"Cache Size "<< Ex_cacheSize[2] <<"\n" ;
        L1CacheSize= Ex_cacheSize[0];
        L2CacheSize= Ex_cacheSize[1];
        L3CacheSize= Ex_cacheSize[2];

    }else
    {
        L1CacheSize=32768, L2CacheSize=1048576, L3CacheSize= 1441792;
    }
    

    float L1_WSS=0, L2_WSS=0, L3_WSS=0, Mem_WSS=0;
    bool Done;
    for (auto it = bandFootprints.rbegin(); it != bandFootprints.rend(); ++it)
    {
        Done = true;
        long int itVal = *it;
        // dbgs() << " bandFootprints: " << itVal << "\n";
        if(itVal < L1CacheSize && L1CacheSize>0 && Done){
            Done = false;
            L1CacheSize -= itVal;
            L1_WSS += itVal;
        }else if(itVal < L2CacheSize && L2CacheSize>0 && Done){
            Done = false;
            L2CacheSize -= itVal;
            L2_WSS += itVal;
        }else if(itVal < L3CacheSize && L3CacheSize>0 && Done){
            Done = false;
            L3CacheSize -= itVal;
            L3_WSS += itVal;
        }else{
            Mem_WSS += itVal;
        }
    }

    if (!explicit_tileSizes.empty() && !explicit_pmaps.empty()){
        // Testing/Printing Working set sizes
        dbgs()<< "L1_WSS "<< L1_WSS << "\n";
        dbgs()<< "L2_WSS "<< L2_WSS << "\n"; 
        dbgs()<< "L3_WSS "<< L3_WSS << "\n"; 
        dbgs()<< "Mem_WSS "<< Mem_WSS << "\n"; 
    }

    //Adding Values in a Dataset.
    std::vector<float> rowDataset;
    long int total_WSS = L1_WSS + L2_WSS + L3_WSS + Mem_WSS;
    rowDataset.push_back(L1_WSS/total_WSS);
    rowDataset.push_back(L2_WSS/total_WSS);
    rowDataset.push_back(L3_WSS/total_WSS);
    rowDataset.push_back(Mem_WSS/total_WSS);
    dataset.push_back(rowDataset); 
}

void PolyDLPass:: generate_ranking(){

    int L1_latency, L2_latency, L3_latency, mem_latency;
    if (!explicit_latencySizes.empty() && explicit_latencySizes.size()==4){
        SmallVector<unsigned, 4> Ex_latencySize;        
        for(auto i : explicit_latencySizes)
            Ex_latencySize.push_back(i);
            
        L1_latency= Ex_latencySize[0];
        L2_latency= Ex_latencySize[1];
        L3_latency= Ex_latencySize[2];
        mem_latency= Ex_latencySize[3];

    }else{
        L1_latency=4, L2_latency=14, L3_latency=60, mem_latency=84;
    }
    for (int i = 0; i < dataset.size(); i++) { 
        dataset[i].push_back(dataset[i][0]*L1_latency + dataset[i][1]*L2_latency + dataset[i][2]*L3_latency + dataset[i][3]*mem_latency);
    }
}

void PolyDLPass:: generateTop5pct(std::vector<SmallVector<unsigned, 6> > tileSizeOut,std::vector<SmallVector<unsigned, 4> > permMapOut){
    // float optProxy = dataset[0][4];
            // int optVersion = 0;
            // std::vector<float> proxyVector;
            for (int i = 0; i < dataset.size(); i++) { 
                dataset[i].push_back(float(i));
            //     proxyVector.push_back(dataset[i][4]);
            //     if(dataset[i][4]< optProxy){
            //         optProxy = dataset[i][4];
            //         optVersion = i;
            //     }

            //     dbgs() << tileSizeOut[i%tileSizeOut.size()][0] << " "; 
            //     dbgs() << tileSizeOut[i%tileSizeOut.size()][1] << " "; 
            //     dbgs() << tileSizeOut[i%tileSizeOut.size()][2] << " "; 
            //     dbgs() << permMapOut[int(i/tileSizeOut.size())][0] << " "; 
            //     dbgs() << permMapOut[int(i/tileSizeOut.size())][1] << " "; 
            //     dbgs() << permMapOut[int(i/tileSizeOut.size())][2] << " "; 
            //     for (int j = 0; j < dataset[i].size(); j++) 
            //         dbgs() << dataset[i][j] << " "; 
            //     dbgs() << "\n"; 
            }
            
            // int minElementIndex = std::min_element(proxyVector.begin(),proxyVector.end()) - proxyVector.begin();
            // float minElement = *std::min_element(proxyVector.begin(), proxyVector.end());
            // dbgs() << "minElementIndex:" << minElementIndex << ", minElement:" << minElement << '\n';

            sort(dataset.begin(), dataset.end(),sortcol); 
            // dbgs()<<"Column se " << dataset[0][5]<< "\n";
            // dbgs() << "Tiles size for optimization " << optVersion; 

            int place_idx;
            for (int i = 0; i < int(dataset.size()*0.05); i++) {
                place_idx = int(dataset[i][5]);
                dbgs() << tileSizeOut[place_idx%tileSizeOut.size()][0] << ","; 
                dbgs() << tileSizeOut[place_idx%tileSizeOut.size()][1] << ","; 
                dbgs() << tileSizeOut[place_idx%tileSizeOut.size()][2] << ","; 
                dbgs() << permMapOut[int(place_idx/tileSizeOut.size())][0] << ","; 
                dbgs() << permMapOut[int(place_idx/tileSizeOut.size())][1] << ","; 
                dbgs() << permMapOut[int(place_idx/tileSizeOut.size())][2] << ",";
               for (int j = 0; j < dataset[i].size() -1; j++) 
                    dbgs() << dataset[i][j] << ","; 
                dbgs() << "\n"; 
            }
}

void PolyDLPass::computeWorkingSetSizes(ArrayRef<AffineForOp> band) {
    LLVM_DEBUG(dbgs() << "In computeWorkingSetSizes()\n ");

   std::vector<long int> bandFootprints;   
   for(unsigned i = 0, e = band.size(); i < e; i++) {
        auto fp = getMemoryFootprintBytes(band[i], 0);
        if (fp) {
            // dbgs() << " Functions Size: " << fp.getValue() << "\n";
            bandFootprints.push_back(fp.getValue());
        } else {
            dbgs() << " fp is NULL \n";
        }
    }

    statsWorkingSetSizes(bandFootprints);
}


void PolyDLPass::generateFuncCopies(FuncOp f, SmallVector<unsigned, 6> tileSizes, SmallVector<unsigned, 4> permMap){
    
    std::vector<SmallVector<AffineForOp, 6>> bands;
    getTileableBands(f, &bands);

    for (auto &band : bands) {
        unsigned loopNestIdx = band.size() - 1;

        SmallVector<AffineForOp, 6> nest;
        std::vector<int>::iterator it;
        int index = 0;
        for (auto i : permMap){
            if (i==0)
                break;
            index++;
        } 
        // for(auto i : band)
        //     dbgs()<< "band" << i << "\n";

        // dbgs() << "band size " << band.size() <<"\n";
        if (band.size() == permMap.size() && band.size() == tileSizes.size()){

            permuteLoops(band, permMap);

            if(auto temp =  dyn_cast<AffineForOp>(*band[index]))
                getPerfectlyNestedLoops(nest, temp);

            SmallVector<AffineForOp, 6> tiledNest;
            if (failed(tilePerfectlyNested(nest, tileSizes, &tiledNest)))
            return signalPassFailure();

            // for (auto i : tiledNest) 
            //     dbgs()<< "tiledNest" << *i  << '\n'; 

            unsigned innermostLoopIdx = tiledNest.size() -1;

            auto innermostLoop = tiledNest[innermostLoopIdx];
            // dbgs()<< "innermostLoop" <<  innermostLoop << '\n';
            auto loopNest = tiledNest[loopNestIdx];

            AffineStoreOp store;
            for (auto &op : *innermostLoop.getBody()) {
            if (auto ld = dyn_cast<AffineStoreOp>(op)) {
                store = ld;
                break;
            }
            }

            AffineCopyOptions copyOptions = {/*generateDma=*/false,
                                        /*slowMemorySpace=*/0,
                                        /*fastMemorySpace=*/0,
                                        /*tagMemorySpace=*/0,
                                        /*fastMemCapacityBytes=*/32 * 1024 * 1024UL};
            DenseSet<Operation *> copyNests;

            // if(store)
            //     affineDataCopyGenerate(loopNest, copyOptions, store.getMemRef(), copyNests);

            computeWorkingSetSizes(tiledNest);   
        }     
    }

}

void PolyDLPass::runOnFunction() {
	LLVM_DEBUG(dbgs() << "Running the PolyDL Recognizer pass \n");
    std::vector<SmallVector<unsigned, 6> > tileSizeOut;
    std::vector<SmallVector<AffineForOp, 6>> bands;
    std::vector<int> lowerBound,upperBound;
    SmallVector<unsigned, 6> out;

    FuncOp f = getFunction();
    getTileableBands(f, &bands);

    for (auto &band : bands) {

        int bandSize = band.size();

        MutableArrayRef<AffineForOp> origLoops = band;
        for (unsigned i = 0; i < band.size(); i++) {
            AffineForOp rootAffineForOp = origLoops[i];

            //Setting Lower bound.
            if(rootAffineForOp.getConstantLowerBound()>=16){
                lowerBound.push_back(rootAffineForOp.getConstantLowerBound());
            }else{
                lowerBound.push_back(16);
            }
            //Setting Upper bound.
            upperBound.push_back(rootAffineForOp.getConstantUpperBound());
        }
        
        // Generating Tile Sizes
        generateTileSizes(0,bandSize,lowerBound,upperBound,out,tileSizeOut); 

        // Generating Permutation Maps
        std::vector<SmallVector<unsigned, 4> > permMapOut;  
        unsigned permMapInitial[bandSize];
        for (int i=0;i<bandSize;++i)
            permMapInitial[i] = i;
        generatePermutationMaps(permMapInitial, bandSize, permMapOut); 
        
        if (!explicit_tileSizes.empty() && !explicit_pmaps.empty()){
            SmallVector<unsigned, 6> Ex_tileSizes;
            SmallVector<unsigned, 4> Ex_permMap;
            
            for(auto i : explicit_tileSizes)
                Ex_tileSizes.push_back(i);
            for(auto i : explicit_pmaps)
                Ex_permMap.push_back(i);
                
            generateFuncCopies(f,Ex_tileSizes,Ex_permMap);

        }else if (band.size()==3){
            for (auto &permMap : permMapOut)
                for (unsigned p = 0; p < tileSizeOut.size(); p++) 
                    generateFuncCopies(f.clone(),tileSizeOut[p],permMap);
            
            generate_ranking();

            generateTop5pct(tileSizeOut,permMapOut);

            generateFuncCopies(f,tileSizeOut[int(dataset[0][5])%tileSizeOut.size()],permMapOut[int(int(dataset[0][5])/tileSizeOut.size())]);

        }

    }

}