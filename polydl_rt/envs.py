import os
import subprocess
import re

class PolyDL_Env():
    def __init__(self, problem_size):
        # data
        self.problems = problem_size

        # instance Lower bounds
        self.lb_O2M = None
        self.lb_O2N = None
        self.lb_O2K = None
        self.lb_O1M = None
        self.lb_O1N = None
        self.lb_O1K = None
        self.lb_Step_M = None
        self.lb_Step_N = None
        self.lb_Step_K = None
        
        # instance Upper bounds
        self.lb_O2M = None
        self.lb_O2N = None
        self.lb_O2K = None
        self.lb_O1M = None
        self.lb_O1N = None
        self.lb_O1K = None
        self.lb_Step_M = None
        self.lb_Step_N = None
        self.lb_Step_K = None
        
        self.GFLOPS = None
        self.cur_step = None
        self.done = None
        
        self.lb_List = None
        self.ub_List = None

        # action space
        self.action_space = [6,7,8,15,16,17,18] #list(range(19))

        self._reset()
    
    def _reset(self):
        
        self.lb_O2M = min(128,self.problems[0])
        self.lb_O2N = min(self.lb_O2M,min(64,self.problems[1]))
        self.lb_O2K = min(self.lb_O2N,min(64,self.problems[2]))
        self.lb_O1M = min(self.lb_O2K,min(32,self.problems[0]))
        self.lb_O1N = min(self.lb_O1M,min(32,self.problems[1]))
        self.lb_O1K = min(self.lb_O1N,min(32,self.problems[2]))
        self.lb_Step_M = 2
        self.lb_Step_N = 16
        self.lb_Step_K = 1
        
        self.ub_O2M = self.problems[0]
        self.ub_O2N = self.problems[1]
        self.ub_O2K = self.problems[2]
        self.ub_O1M = self.problems[0]
        self.ub_O1N = self.problems[1]
        self.ub_O1K = self.problems[2]
        self.ub_Step_M = min(64,self.ub_O1M)
        self.ub_Step_N = min(128,self.ub_O1N)
        self.ub_Step_K = min(64,self.ub_O1K)
        
        self.GFLOPS = 50
        self.reward = 0
        self.done = False
        self.cur_step = [self.lb_O2M,
                        self.lb_O2N,
                        self.lb_O2K,
                        self.lb_O1M,
                        self.lb_O1N,
                        self.lb_O1K,
                        self.lb_Step_M,
                        self.lb_Step_N,
                        self.lb_Step_K]
        
        self.lb_List = [self.lb_O2M,
                        self.lb_O2N,
                        self.lb_O2K,
                        self.lb_O1M,
                        self.lb_O1N,
                        self.lb_O1K,
                        self.lb_Step_M,
                        self.lb_Step_N,
                        self.lb_Step_K]
        
        self.ub_List = [self.ub_O2M,
                        self.ub_O2N,
                        self.ub_O2K,
                        self.ub_O1M,
                        self.ub_O1N,
                        self.ub_O1K,
                        self.ub_Step_M,
                        self.ub_Step_N,
                        self.ub_Step_K]
        
    
    def _get_state(self):
        _state = []
        for i in self.cur_step:
            _state.append(i)
        return _state
    
    def runCMd(self,cmd):
        process = subprocess.Popen(cmd, stdout = subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout
    
    def _step(self,action):
        current_state = self._get_state()
        # new_state = [0]*19
        print("action ",action)
        idx = self.action_space[action]
        
        if idx == 18:
            return current_state, self.reward, self.done
        
        if idx > 8:
            current_state[idx%9]= max( self.lb_List[idx%9], int(current_state[idx%9]/2))
        else:
            current_state[idx] = min ( self.ub_List[idx], current_state[idx] *2)
        
        print(current_state)
        # Run JIT Compiler
        cmd=['sh','run_with_jit_compiler_only_avx.sh']
        for val in current_state:
            cmd.append(str(val))
        
        # out = self.runCMd(['sh','run_with_jit_compiler.sh','128','256','256','64','64','64','2','64','1'])
        out = self.runCMd(cmd)

        y = re.search("GFLOPS=.*\d",str(out))
        if y:
            x=y[0].split("=")
            print(float(x[1]))
            self.reward = ((float(x[1]) - self.GFLOPS)/ self.GFLOPS)*10
            print(self.reward)
            self.GFLOPS = float(x[1])

        self.cur_step = current_state
        return self.cur_step, self.reward, self.done
    
    def _currentGFlops(self):
        return self.GFLOPS