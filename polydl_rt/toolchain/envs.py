import os
import subprocess
import re
import pandas as pd
import numpy as np

# train_df = pd.read_csv("data.csv",header=None)
# train_df = np.array(train_df)

class PolyDL_Env():
    def __init__(self, problem_size,file):
        # data
        self.problems = problem_size

        # Load Dataset
        # fileName = ""
        # for val in problem_size:
        #     fileName += str(val) + "_"
        # fileName = fileName[:-1]      
        # fileName+=".csv" 
        self.train_df = pd.read_csv(file,header=None)
        # self.train_df = np.array(self.train_df)

        t=file.split('_')
        s1=""
        for val in t[4:7]:
            s1+=val+"_"
        self.make_cmd = s1[:-1]

        # instance Lower bounds
        self.lb_OM = None
        self.lb_ON = None
        self.lb_OK = None
        self.lb_Step_M = None
        self.lb_Step_N = None
        self.lb_Step_K = None
        
        # instance Upper bounds
        self.ub_OM = None
        self.ub_ON = None
        self.ub_OK = None
        self.ub_Step_M = None
        self.ub_Step_N = None
        self.ub_Step_K = None
        
        self.GFLOPS = None
        self.MaxFLOPS = None
        self.cur_step = None
        self.done = None
        
        self.lb_List = None
        self.ub_List = None

        # action space
        self.action_space = list(range(7)) #[6,7,8,15,16,17,18]

        self._reset()
    
    def _reset(self):
        
        self.lb_OM = self.problems[3]
        self.lb_ON = self.problems[4]
        self.lb_OK = self.problems[5]
        self.lb_Step_M = 1
        self.lb_Step_N = 16
        self.lb_Step_K = 1
        
        self.ub_OM = self.problems[3]
        self.ub_ON = self.problems[4]
        self.ub_OK = self.problems[5]
        self.ub_Step_M = min(32,self.ub_OM)
        self.ub_Step_N = min(32,self.ub_ON)
        self.ub_Step_K = min(32,self.ub_OK)
        
        self.GFLOPS = self.train_df[3].min()
        self.MaxFLOPS = 0
        self.reward = 0
        self.done = False
        self.cur_step = [4,#self.lb_Step_M,
                        self.lb_Step_N,
                        4,#self.lb_Step_K,
                        self.lb_OM,
                        self.lb_ON,
                        self.lb_OK
                        ]
        
        self.lb_List = [self.lb_Step_M,
                        self.lb_Step_N,
                        self.lb_Step_K,
                        self.lb_OM,
                        self.lb_ON,
                        self.lb_OK
                        ]
        
        self.ub_List = [self.ub_Step_M,
                        self.ub_Step_N,
                        self.ub_Step_K,
                        self.ub_OM,
                        self.ub_ON,
                        self.ub_OK
                        ]
        
    
    def _get_state(self):
        _state = []
        for i in self.cur_step:
            _state.append(i)
        return _state
    
    def runCMd(self,cmd):
        process = subprocess.Popen(cmd, stdout = subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout
    
    def _getResult(self,current_state):
        cmd= self.make_cmd + "__"
        self.problems[3]
        for val in self.problems:
            cmd+=str(val) + "_"
        for val in current_state[:3]:
            cmd+=str(val) + "_"

        result = self.train_df[self.train_df[0].str.contains(cmd)]
        return result
    
    def _step(self,action):
        current_state = self._get_state()
        # new_state = [0]*19
        print("action ",action)
        idx = self.action_space[action]
        
        if idx == 6:
            # return current_state, self.reward, self.done
            result = self._getResult(current_state)
        elif idx > 2:
            while True:
                current_state[idx%3]= max( self.lb_List[idx%3], int(current_state[idx%3]/2))
                result = self._getResult(current_state)
                if(len(result)>0) or current_state[idx%3] == self.lb_List[idx%3]:
                    break

        else:
            while True:
                current_state[idx] = min ( self.ub_List[idx], current_state[idx] *2)
                result = self._getResult(current_state)
                if(len(result)>0) or current_state[idx] == self.ub_List[idx]:
                    break
        
        # print(current_state)
        # # Run JIT Compiler
        # cmd=['sh','run_with_jit_compiler_only_avx.sh']
        # for val in current_state:
        #     cmd.append(str(val))
        
        # # out = self.runCMd(['sh','run_with_jit_compiler.sh','128','256','256','64','64','64','2','64','1'])
        # out = self.runCMd(cmd)

        # y = re.search("GFLOPS=.*\d",str(out))
        # if y:
        #     x=y[0].split("=")
        #     print(float(x[1]))
        #     self.reward = ((float(x[1]) - self.GFLOPS)/ self.GFLOPS)*10
        #     print(self.reward)
        #     self.GFLOPS = float(x[1])

        # self.cur_step = current_state
        # return self.cur_step, self.reward, self.done
        
        # Use Pre-calculated data set

        # result = self._getResult(current_state)

        self.cur_step = current_state

        print("Curr State ", self.cur_step)

        # result = self.train_df[self.train_df[:,0]==cmd]
        # print(result)
        if len(result) == 0:
            # print("Am i reaching here")
            return self.cur_step, -0.5, self.done        

        result = result.iloc[0]

        self.reward = ((result[3] - self.GFLOPS)/ self.GFLOPS) - 1
        # print("reward ", self.reward)
        self.GFLOPS = result[3]
        # print("GFlops ",self.GFLOPS)    

        if self.GFLOPS > self.MaxFLOPS:
            self.MaxFLOPS = self.GFLOPS
        
        return self.cur_step, self.reward if idx!=6 else 0 , self.done


    
    def _currentGFlops(self):
        return self.GFLOPS
    
    def _MaxGFlops(self):
        return self.MaxFLOPS