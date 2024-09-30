from enviroment import Maze
import numpy as np
import heapq
from enum import Enum

class SearchAgent():
    
    class Solver(Enum):
        DFS = 1,
        BFS = 2,
        ASTAR = 3,
        B_A_B = 4

    def __init__(self,env:Maze):
        self.env = env
        self.percepts = env.initial_percepts()
        self.F = [[self.percepts['pos']]]
        s = self.percepts['pos'] # starting position
        self.C = [env.map[s[0]][s[1]]]
        self.H = [self.Manhatan(env.map.shape[0],env.map.shape[1],s[0],s[1])]
        self.visited = [self.percepts['pos']]

    def Manhatan(self,x,y,x_atual,y_atual)->int:
        return (x-x_atual) + (y-y_atual)

    def add_Stack(self,F,path,n):
        F.append(path+n)
    def select_Stack(self,F):
        return F.pop(-1)
    
    def add_Queue(self,F,path,n):
        F.append(path+n)
    def select_Queue(self,F):
        return F.pop(0)
    
    def add_priority(self,F,path,n,priority):
        heapq.heappush(F,(priority,path+n))

    def select_priority(self,F):
        return heapq.heappop(F)[1]
    
    
    
    def solve(self,select_fn,add_fn):
        while(self.F):
            path = select_fn(self.F)
            c = select_fn(self.C) 

            action = {'path': path, 'move_to': path[-1]}

            self.percepts = self.env.state_transition(action)
            
            if (path[-1] == self.percepts['exit']).all():
                return path, c
            
            for n,cn in zip(self.percepts['neighbors'],self.percepts['neighbors_cost']):
                if not(any(np.array_equal(n,p) for p in path)) and not(any(np.array_equal(n,p) for p in self.visited)) :
                    add_fn(self.F, path, [n])
                    add_fn(self.C, c, cn)
                    self.visited.append(n)
        
        return None,None
    
    def solve_astar(self):
        selected_path = self.F.pop()
        selected_cost = self.C.pop()
        self.add_priority(self.F,selected_path,[],0)
        self.add_priority(self.C,selected_cost,0,0)
        while(self.F):
            path = self.select_priority(self.F)
            c = self.select_priority(self.C) 

            action = {'path': path, 'move_to': path[-1]}

            self.percepts = self.env.state_transition(action)
            
            if (path[-1] == self.percepts['exit']).all():
                return path, c
            
            for n,cn,hn in zip(self.percepts['neighbors'],self.percepts['neighbors_cost'],self.percepts['neighbors_heuristic']):
                if not(any(np.array_equal(n,p) for p in path)) and not(any(np.array_equal(n,p) for p in self.visited)) :
                    self.add_priority(self.F, path, [n],cn+hn)
                    self.add_priority(self.C, c, cn,cn+hn)
                    self.visited.append(n)
        
        return None,None
    
    def solve_branch_and_bound(self):
        best_path,bound = self.solve(select_fn=self.select_Stack,add_fn=self.add_Stack)
        
        self.env = env
        self.percepts = env.initial_percepts()
        self.F = [[self.percepts['pos']]]
        s = self.percepts['pos'] # starting position
        self.C = [env.map[s[0]][s[1]]]
        self.H = [self.Manhatan(env.map.shape[0],env.map.shape[1],s[0],s[1])]
        self.visited = [self.percepts['pos']]
        
        selected_path = self.F.pop()
        selected_cost = self.C.pop()
        self.add_priority(self.F,selected_path,[],0)
        self.add_priority(self.C,selected_cost,0,0)

        while(self.F):
            path = self.select_priority(self.F)
            c = self.select_priority(self.C) 

            action = {'path': path, 'move_to': path[-1]}

            self.percepts = self.env.state_transition(action)
            
            if (path[-1] == self.percepts['exit']).all():
                best_path = path
                bound = c
            
            for n,cn,hn in zip(self.percepts['neighbors'],self.percepts['neighbors_cost'],self.percepts['neighbors_heuristic']):
                if not(any(np.array_equal(n,p) for p in path)) and not(any(np.array_equal(n,p) for p in self.visited)) and (cn + hn) < bound:
                    self.add_priority(self.F, path, [n],cn+hn)
                    self.add_priority(self.C, c, cn,cn+hn)
                    self.visited.append(n)
        
        return best_path,bound

    def act(self,solver:Solver):
        path = None
        cost = None
        if(solver.name == 'DFS'):
            path, cost = self.solve(select_fn=self.select_Stack,add_fn=self.add_Stack)
        else:
            if(solver.name == 'BFS'):
                path, cost = self.solve(select_fn=self.select_Queue,add_fn=self.add_Queue)
            else:
                if(solver.name == 'ASTAR'):
                    path, cost = self.solve_astar()
                else:
                    if(solver.name == 'B_A_B'):
                        path,cost = self.solve_branch_and_bound()
        
        return path,cost


if __name__ == '__main__':
    nrow = 15
    ncol = 15
    env = Maze(nrow,ncol,[0,0],[nrow-1,ncol-1],pobs=0.2)

    agent1 = SearchAgent(env)
    agent2 = SearchAgent(env)
   
    path1, cost1 = agent1.act(SearchAgent.Solver.ASTAR)
    path2, cost2 = agent2.act(SearchAgent.Solver.B_A_B)
    
    print(path1)
    print(f'Cost: {cost1}')
    print('--------------------------------------')
    print(path2)
    print(f'Cost: {cost2}')


