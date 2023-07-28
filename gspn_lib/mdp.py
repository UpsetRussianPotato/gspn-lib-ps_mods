#! /usr/bin/env python3
import os
from gspn_lib import gspn as pn
from gspn_lib import gspn_tools as pn_tools
from gspn_lib import gspn_analysis as pn_analysis
from graphviz import Digraph

GOAL_REWARD = 1
STD_REWARD = -0.01
THRESHOLD = 4e-10


#######################################################################
###                             MDP CLASSES                         ###
#######################################################################

class StateAction(object):
    def __init__(self):
        self.act_name = [""]    
        self.act_edges = {}     #{"resulting_state" : ["rate/probability", "cost/reward"]}

    def set_state_name(self, name):
        if type(name) == str:
            self.act_name = [name]
        else:
            print("error: state action name data")
    
    def update_state_action(self, action_name, next_state_name, rate_prob, cost_rwd):
        self.set_state_name(action_name)
        self.act_edges.update({next_state_name:[rate_prob, cost_rwd]})
        #! DOES IT MAKE SENSE to have a action reward????


class MDPState(object):
    def __init__(self):
        self.state_name = [""]
        self.state_type = [""]        # ["vanishing"],["tangible"], ["dead"]
        self.state_policy = [""]
        self.state_reward = 0.0
        self.state_value = 0.0
        self.state_actions = {}     # dict of StateAction()
        self.total_freq_map = {}
        self.exit_rate = 0.0
        #! DOES IT MAKE SENSE to have a action reward????

    def set_state_name(self, name):
        if type(name) == str:
            self.state_name = [name]
        else:
            print("error: mdp state name data")

    def set_state_type(self, s_type):
        if type(s_type) == str:
            self.state_type = [s_type]
        else:
            print("error: mdp state type data")
    
    def set_state_reward(self, rwd):
        if type(rwd) == float:
            self.state_reward = rwd
        else:
            print("error: mdp state reward data")

    def update_edge(self, action_name, next_state_name,  rate_prob, cost_rwd):
        existing_act = self.state_actions.get(action_name)
        #Add new action
        if existing_act == None:
            new_act = StateAction()
            self.state_actions.update({action_name:new_act})
            new_act.update_state_action(action_name, next_state_name, rate_prob, cost_rwd)
            pass
        #Add edge
        else:
            existing_act.update_state_action(action_name, next_state_name, rate_prob, cost_rwd)
    
    def remove_edge(self, action_name, next_state_name):
        state_act = self.state_actions.get(action_name)
        
        #Remove state action edge
        if list(state_act) > 1 and next_state_name != None:
            state_act.pop(next_state_name)
        #Remove state action
        else:
            self.state_actions.pop(action_name)
    
    def calculate_total_frequency(self):
        if self.state_type[0] == 'T':
            self.total_freq_map.clear()
            for action_name in self.state_actions:
                action = self.state_actions.get(action_name)
                for resulting_state in action.act_edges:
                    edge_data = action.act_edges.get(resulting_state)
                    new_rate = edge_data[0]
                    rate = self.total_freq_map.get(resulting_state)
                    if rate == None:
                        rate = 0
                    rate += new_rate
                    self.total_freq_map.update({resulting_state:rate})
    
    def calculate_exit_frequency(self):
        if self.state_type[0] == 'T':
            self.exit_rate = 0
            for state_id in self.total_freq_map:
                if state_id != self.state_name[0]:
                    m_to_mm_rate = self.total_freq_map.get(state_id)
                    self.exit_rate += m_to_mm_rate


class MDP(object):
    def __init__(self):
        self.nodes = {}                     # dict of {id : MDPState()} #self.edges = []
        self.mdp_state_mapping = {}
        self.ideal_state_path = None
        self.max_exit_rate = 0

        self.tangible_states = []
        self.vanishing_states = []
        self.deadlock_states = []
    
    def add_new_state(self, state_id, state_reward=0.0, state_type="T"):
        new_state = MDPState()
        self.tangible_states.append(state_id)
        self.nodes.update({state_id: new_state})

        new_state.set_state_type(state_type)
        new_state.set_state_reward(state_reward)
        new_state.set_state_name(state_id)
        #self.mdp_state_mapping.update({state_id: [next_marking, new_state.state_type]})

    def generate_from_gspn(self, gspn):
        self.__gspn = gspn
        self.nodes = {}
        self.mdp_state_mapping = {}
        self.deadlock_free = True
        self.ideal_state_path = None
        self.max_exit_rate = 0

        self.tangible_states = []
        self.vanishing_states = []
        self.deadlock_states = []

        # obtain the enabled transitions for the initial marking
        exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()
        # obtain the initial marking
        next_marking__dict = self.__gspn.get_initial_marking()

        # initial state name
        marking_index = 0
        next_state_id = 'S' + str(marking_index)
        next_marking = []
        for place_id, ntokens in next_marking__dict.items():
            next_marking.append(ntokens)

        state_stack = []
        # initialization of the state stack with initial state
        self.__add_new_state_from_gspn(state_stack, next_state_id, next_marking)

        while state_stack:
            # pop a marking from the stack using a FIFO methodology
            #state_stack.reverse()
            state_info = state_stack.pop(0)
            #state_stack.reverse()

            state_id = state_info[0]
            current_marking = state_info[1]
            
            # set the current marking as the marking of the GSPN
            self.__gspn.set_marking_list(current_marking)
            current_state = self.nodes.get(state_id)
            [enabled_transitions, transition_type] = self.__get_gspn_enabled_transition()
            
            if enabled_transitions:
                # sum the rates from all enabled transitions, to obtain the transition probabilities between markings
                rate_sum = sum(enabled_transitions.values())
                for trans, rate in enabled_transitions.items():
                    # for each enabled transition of the current marking fire it to land in a new marking
                    self.__gspn.fire_transition(trans)

                    # get the new marking where it landed
                    next_marking = self.__gspn.get_current_marking_list()
                    self.__check_boundness(next_marking)

                    # check if the marking was already added as a node or not
                    marking_already_exists = False
                    for state_id_x, state in self.mdp_state_mapping.items():
                        marking = state[0]
                        if next_marking == marking:
                            marking_already_exists = True
                            next_state_id = state_id_x
                            break

                    if not marking_already_exists:
                        marking_index += 1
                        next_state_id = 'S' + str(marking_index)
                        # add new state
                        self.__add_new_state_from_gspn(state_stack, next_state_id, next_marking)

                    if current_state.state_type[0] == 'V':
                        current_state.update_edge(trans, next_state_id, 1, 0)
                        #Immediate transitions are assumed to be executed instantaneously and without uncertainty
                    if current_state.state_type[0] == 'T':
                        current_state.update_edge(trans, next_state_id, rate, 0)

                    # revert the current marking
                    self.__gspn.set_marking_list(current_marking)
                if current_state.state_type[0] == 'T':
                    wait_edge_name = str("wait" + state_id)
                    current_state.update_edge(wait_edge_name, state_id, -rate_sum, 0)

        self.__calculate_max_exit_rate()
        self.__calculate_updated_tangible_states_rates()
        self.value_iteration()
        self.get_policy()

        [self.ideal_state_path, action_list]  = self.__get_simple_gspn_plan()
        
        return

    def __get_gspn_enabled_transition(self):
        # obtain the enabled transitions for current marking
        exp_transitions_en, immediate_transitions_en = self.__gspn.get_enabled_transitions()
        if immediate_transitions_en:
            enabled_transitions = immediate_transitions_en.copy()
            transition_type = 'I'
        elif exp_transitions_en:
            enabled_transitions = exp_transitions_en.copy()
            transition_type = 'E'
        else:
            enabled_transitions = {}
            transition_type = None
        return [enabled_transitions, transition_type]

    def __check_boundness(self, eval_marking):
        for state_id, state in self.mdp_state_mapping.items():
            # checks if the new marking is unbounded
            unbounded_state = True
            marking = state[0]
            for place_mark in marking:
                indexy = marking.index(place_mark)
                eval_mark = eval_marking[indexy]
                if  eval_mark <= place_mark:
                    unbounded_state = False
            if unbounded_state:
                print("erro: unbounded states...")
                exit()

    def __get_simple_gspn_plan(self):
        starting_state = list(self.nodes)[0]
        action_list = []
        ideal_states_path = []

        ideal_states_path.append(starting_state)
        state = self.nodes.get(starting_state)
        while state != None and state.state_type[0] != 'D':
            desired_act = state.state_policy[0]
            action_list.append(desired_act)
            desired_state_act = state.state_actions.get(desired_act)
            for resulting_state_name in desired_state_act.act_edges:
                prob = 0.0
                possible_next_state = self.nodes.get(resulting_state_name)
                edge_data = desired_state_act.act_edges.get(resulting_state_name)
                if prob < edge_data[0]:
                    if possible_next_state.state_name != state.state_name:
                        next_state_name = resulting_state_name
                        prob = edge_data[0]
            state = self.nodes.get(next_state_name)
            ideal_states_path.append(next_state_name)

        return [ideal_states_path, action_list]

    def __add_new_state_from_gspn(self, state_stack, next_state_id, next_marking):
        new_state = MDPState()
        new_state.set_state_name(next_state_id)
        self.nodes.update({next_state_id: new_state})

        self.__gspn.set_marking_list(next_marking)
        [enabled_transitions, transition_type] = self.__get_gspn_enabled_transition()

        if transition_type == 'I':
            new_state.set_state_type('V')  # vanishing marking
            self.vanishing_states.append(next_state_id)
            new_state.state_reward = STD_REWARD
        elif transition_type == 'E':
            new_state.set_state_type('T')  # tangible marking
            self.tangible_states.append(next_state_id)
            new_state.state_reward = STD_REWARD
        else:
            new_state.set_state_type('D')  # deadlock and tangible marking
            self.deadlock_states.append(next_state_id)
            self.deadlock_free = False
            new_state.state_reward = GOAL_REWARD

        self.mdp_state_mapping.update({next_state_id: [next_marking, new_state.state_type]})
        state_stack.append([next_state_id, next_marking])
    
    def __calculate_updated_tangible_states_rates(self):
        g_eta = self.max_exit_rate + 1
        for state_name in self.tangible_states:
            state = self.nodes.get(state_name)
            for action_name in state.state_actions:
                action = state.state_actions.get(action_name)
                for resulting_state in action.act_edges:
                    edge_data = action.act_edges.get(resulting_state)
                    rate = edge_data[0]
                    rate = rate/g_eta
                    if state_name == resulting_state:
                        rate = 1 + rate
                    cost_rwd = edge_data[1]
                    state.update_edge(action_name, resulting_state, rate, cost_rwd)
                    state.total_freq_map.update({resulting_state:rate})
    
    def __calculate_max_exit_rate(self):
        self.max_exit_rate = 0
        for state_id in self.nodes:
            self.nodes.get(state_id).calculate_total_frequency()
            self.nodes.get(state_id).calculate_exit_frequency()
            if self.max_exit_rate < self.nodes.get(state_id).exit_rate:
                self.max_exit_rate = self.nodes.get(state_id).exit_rate
    
    def utility(self, prob_next_s, value):
        return prob_next_s*value
    
    def value_iteration(self):
        i = 0
        not_stop = True
        self.next_iteration_values__dict = {}
        while not_stop:
            i +=1
            delta = 0.0
            #start at terminal state for better performance
            for state_name in self.nodes:
                utility = 0.0
                state = self.nodes.get(state_name)
                prev_value = state.state_value
                for action_name in state.state_actions:
                    state_act = state.state_actions.get(action_name)
                    new_utility = 0.0
                    for resulting_state in state_act.act_edges:
                        edge_data = state_act.act_edges.get(resulting_state)
                        prev_value_n_state = self.nodes.get(resulting_state).state_value
                        new_utility += self.utility(edge_data[0], prev_value_n_state)
                        if new_utility > utility:
                            utility = new_utility
                value = state.state_reward + utility
                
                self.next_iteration_values__dict.update({state_name:value})
                if delta < abs(prev_value-value):
                    delta = abs(prev_value-value)

            for state_name in self.nodes:
                state = self.nodes.get(state_name)
                state.state_value = self.next_iteration_values__dict.get(state_name)
            if delta < THRESHOLD:
                not_stop = False

    def get_policy(self):
        #arg max a of the sum of the multiplication of the resulting states of a and s with its value
        for state_name in self.nodes:
            state = self.nodes.get(state_name)
            state.state_policy = [""]
            max_action_value = -GOAL_REWARD
            for action_name in state.state_actions:
                state_act = state.state_actions.get(action_name)
                action_value = 0.0
                for resulting_state_name in state_act.act_edges:
                    edge_data = state_act.act_edges.get(resulting_state_name)
                    resulting_state = self.nodes.get(resulting_state_name)
                    rate = edge_data[0]
                    action_value += rate*resulting_state.state_value
                if max_action_value < action_value:
                    max_action_value = action_value
                    state.state_policy = [action_name]

    def __draw_states_subset(self, diagraph, state_subset, node_line_color):
        for state_name in state_subset:
            state = self.nodes.get(state_name)
            diagraph.node(state.state_name[0], shape='doublecircle', label=state.state_name[0], height='0.6', width='0.6', fixedsize='true', color=node_line_color)
            for action_name in state.state_actions:
                action = state.state_actions.get(action_name)
                for resulting_state in action.act_edges:
                    edge_data = action.act_edges.get(resulting_state)
                    if state.state_policy[0] == action_name:
                        edge_label = "Act: " + str(action_name) + " - Data: " + str(round(edge_data[0],2)) + ' (' + str(round(edge_data[1],2)) + ')'
                        diagraph.edge(state_name, resulting_state, label=edge_label, color="hotpink")
                    else:
                        edge_label = "Act: " + str(action_name) + " - Data: " + str(round(edge_data[0],2)) + ' (' + str(round(edge_data[1],2)) + ')'
                        diagraph.edge(state_name, resulting_state, label=edge_label)

    def __draw_policy_plan(self, diagraph, fill_color, ideal_states_path):
        for state_name in ideal_states_path:
            state = self.nodes.get(state_name)
            if state_name == list(self.nodes)[0] or state.state_type[0] == 'D':
                diagraph.node(state.state_name[0], shape='doublecircle', label=state.state_name[0], height='0.6', width='0.6', fixedsize='true', fillcolor="green", style="filled")
            else:
                diagraph.node(state.state_name[0], shape='doublecircle', label=state.state_name[0], height='0.6', width='0.6', fixedsize='true', fillcolor=fill_color, style="filled")

    def draw_mdp(self, file='mdp_default', show=True):
        mdp_draw = Digraph(engine='dot')
        mdp_draw.attr('node', forcelabels='true')

        self.__draw_states_subset(mdp_draw, self.vanishing_states, "black")
        self.__draw_states_subset(mdp_draw, self.tangible_states, "blue")
        self.__draw_states_subset(mdp_draw, self.deadlock_states, "red")

        if self.ideal_state_path != None:
            self.__draw_policy_plan(mdp_draw, "lightgreen", self.ideal_state_path)
        mdp_draw.render(file + '.gv', view=show)

        return mdp_draw

    def get_state_policy(self, state_name=None, pn_marking=None):
        '''
        Supply a state identifier - state name (str), simple pn marking (list, e.g. [0, 1, 0]) or a gspn lib marking (dict)
        and get the associated policy 
        '''
        if state_name != None:
            pass
        elif pn_marking != None:
            for searched_state_name, state_data in self.mdp_state_mapping.items():
                if pn_marking == state_data[0]:
                    state_name = searched_state_name
                    break
        else:
            pass
        
        if state_name != None:
            state = self.nodes.get(state_name)
            if state != None:
                policy = state.state_policy
                if len(policy)>0 and policy is not None:
                    return policy[0]

        return None


#######################################################################
###                        AUXILIAR FUNCTIONS                       ###
#######################################################################

def gspn_example(mode="complex"):
    my_gspn = pn.GSPN()

    desired_rate = 50
    g_eta = 0.025
    rate_f = (desired_rate)*g_eta

    desired_rate_ = 15
    rate__ = (desired_rate_)*g_eta

    desired_rate__ = 20
    rate_ = (desired_rate__)*g_eta
    
    arc_in = {}
    arc_out = {}

    if mode == "simple" or mode == "complex":
        places = my_gspn.add_places(['P0', 'P1', 'P2'], [0, 0, 0])
        trans = my_gspn.add_transitions(['t01', 't02'], 
                                        ['imm', 'imm'], 
                                        [rate_f, rate_f])

        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't01', 'P0', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't01', 'P1', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't02', 'P0', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't02', 'P2', 1, True)
        
        if mode == "simple":
            mark = [0, 1]
        else:
            mark = [0, 0]

        places = my_gspn.add_places(['P3', 'P4'], mark)
        trans = my_gspn.add_transitions(['t13', 't31', 't24', 't42', 't34', 't43'], 
                                        [ 'exp', 'exp', 'exp', 'exp', 'exp', 'exp'], 
                                        [rate__, rate__, rate__, rate__, rate_, rate_])

        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't13', 'P1', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't13', 'P3', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't31', 'P3', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't31', 'P1', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't24', 'P2', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't24', 'P4', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't42', 'P4', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't42', 'P2', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't34', 'P3', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't34', 'P4', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't43', 'P4', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't43', 'P3', 1, True)
    
    if mode == "complex":
        places = my_gspn.add_places(['P5', 'P6'], [0, 1])
        trans = my_gspn.add_transitions(['t35', 't53', 't46', 't64', 't56', 't65'], 
                                        ['exp', 'exp', 'exp', 'exp', 'exp', 'exp'], 
                                        [rate__, rate__, rate__, rate__, rate_, rate_])

        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't35', 'P3', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't35', 'P5', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't53', 'P5', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't53', 'P3', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't46', 'P4', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't46', 'P6', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't64', 'P6', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't64', 'P4', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't56', 'P5', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't56', 'P6', 1, True)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't65', 'P6', 1, False)
        my_gspn.create_place_to_transition_arc(arc_in, arc_out, 't65', 'P5', 1, True)

    my_gspn.add_arcs(arc_in, arc_out)

    tools = pn_tools.GSPNtools()
    coverability_tree = pn_analysis.CoverabilityTree(my_gspn)
    coverability_tree.generate()
    tools.draw_gspn(my_gspn, 'run_rvary4', show=True)
    print("Reachablility:\t", len(coverability_tree.nodes))

    mdp = MDP()
    mdp.generate_from_gspn(my_gspn)
    print("States:\t", len(mdp.nodes))
    for state_name in mdp.nodes:
        marking = mdp.mdp_state_mapping.get(state_name)[0]
        state_value = mdp.next_iteration_values__dict.get(state_name)
        state_politika = mdp.get_state_policy(state_name=None, pn_marking=marking)
        print(state_name + str(marking) + ":\t has value %.2f" %state_value + " and its policy is firing " + str(state_politika))
    mdp.draw_mdp()

def mdp_example():
    mdp = MDP()
    state_id_1 = ["S0"]
    edges_1 = ["act1", "S1", 0.95, -0.1]
    state_id_2 = ["S1"]
    edges_2 = ["act2", "S0", 0.95, -0.1]

    states_data = [[state_id_1, [edges_1]], [state_id_2, [edges_2]]]
    for state_data in states_data:
        state_intrinsic_data = state_data[0]
        state_extrinsic_data = state_data[1]
        mdp.add_new_state(state_id=state_intrinsic_data[0])

        for edge_data in state_extrinsic_data:
            action_name = edge_data[0]
            resulting_state_name = edge_data[1]
            rate = edge_data[2]
            action_cost = edge_data[3]
            state = mdp.nodes.get(state_intrinsic_data[0])
            state.update_edge(action_name, resulting_state_name, rate, action_cost)
    # mdp.value_iteration()
    # mdp.get_policy()
    mdp.draw_mdp()

    #absorbing state, decision state, probabilistic state
    #A* and Djistra assume actions are purely deterministic!
    #heuristic search algorithm
    #objective function - specifications beyond the average value
    #augmented state space - consider time passed in the current decision process



if __name__ == "__main__":
    gspn_example()
    #mdp_example()