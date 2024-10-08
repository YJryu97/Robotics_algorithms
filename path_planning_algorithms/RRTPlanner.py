from RRTTree import RRTTree
import numpy as np
import sys
import time


class RRTPlanner(object):
    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config, goal_config):
        """ Some utility functions you can use:
            From RRTTree:
                new_vertex_id = self.tree.AddVertex(node, cost))
                self.tree.AddEdge(start_id, end_id)
                vertex_id, vertex = self.tree.GetNearestVertex(state)
            From RRTPlanner:
                new_state = self.sample(target_state)
                new_state = self.extend(state1, state2)
            From ArmEnvironment:
                cost = self.env.compute_distance(state1, state2
                reached_goal = self.env.goal_criterion(state)
        """
        # Initialize an empty plan.
        goal_id = -1
        plan = []
        plan_time = time.time()

        initial_id = self.tree.AddVertex(start_config, cost=0)
        # TODO: YOUR IMPLEMENTATION STARTS HERE
        for _ in range(self.max_iter):
            x_rand = self.sample(goal_config)
            # self.tree.AddVertex(x_rand, cost=0)
            # new_state = self.sample(x_rand)
            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)
            #print(x_near)
            #self.tree.AddVertex(x_rand, cost=self.env.compute_distance(x_near, goal_config))
            x_new = self.extend(x_near, x_rand)
            #print(x_new)
            if x_new is None:
                continue
            xid = self.tree.AddVertex(x_new, cost=self.env.compute_distance(x_new, goal_config))
            self.tree.AddEdge(x_near_id, xid)

            if self.env.goal_criterion(x_new):
                goal_id = xid
                break
        # YOUR IMPLEMENTATION ENDS HERE            

        if goal_id < 0:
            print("WARNING: RRT Failed!")
            sys.exit(-1)

        # Construct plan
        plan.insert(0, goal_config)
        cid = goal_id
        root_id = self.tree.GetRootID()
        while cid != root_id:
            cid = self.tree.edges[cid]
            plan.insert(0, self.tree.vertices[cid])
        plan = np.concatenate(plan, axis=1)

        plan_time = time.time() - plan_time
        print("Planning complete!")
        print(f"Planning Time: {plan_time:.3f}s")

        return plan.T

    def extend(self, x_near, x_rand):
        """
        may be useful:
            self.eta: ratio to extend from x_near to x_rand
            bool_result = self.env.edge_validity_checker(x_near, x_new)
        return None if edge not valid
        """
        delta = x_rand - x_near
        dist = self.env.compute_distance(x_near, x_rand)
    
        # Scale the direction vector(x_near --> x_rand) into self.eta
        if dist > self.eta:
            delta = self.eta * delta

        # x_new
        x_new = x_near + delta

        # Check if the edge between x_near and x_new is valid
        if self.env.edge_validity_checker(x_near, x_new):
            return x_new
        else:
            return None
        
        # YOUR IMPLEMENTATION ENDS HERE

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal
        return self.env.sample()
