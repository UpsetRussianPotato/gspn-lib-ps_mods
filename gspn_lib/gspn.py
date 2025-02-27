#! /usr/bin/env python3
import time
import numpy as np
from gspn_lib import gspn_analysis
import sparse

class GSPN(object):
    """
    """

    def __init__(self):
        """

        """
        self.__sparse_marking = {}
        self.__places = {}
        self.__initial_marking = {}
        self.__initial_marking_sparse = {}
        self.__transitions = {}
        self.__imm_transitions = {}
        self.__imm_transitions_generated = False
        self.__timed_transitions = {}
        self.__timed_transitions_generated = False
        self.__arc_in_m = sparse.COO([[], []], [], shape=(0, 0))
        self.__arc_out_m = sparse.COO([[], []], [], shape=(0, 0))
        self.__ct_tree = None
        self.__ctmc = None
        self.__ctmc_steady_state = None
        self.__ct_ctmc_generated = False
        self.__nsamples = {}
        self.__sum_samples = {}

        # Mappings that translate x_to_y
        self.places_to_index = {}
        self.transitions_to_index = {}
        self.index_to_places = {}
        self.index_to_transitions = {}

    def get_arc_in_m(self):
        return self.__arc_in_m.copy()

    def get_arc_out_m(self):
        return self.__arc_out_m.copy()

    def get_places(self):
        return self.__places.copy()

    def set_places(self, new_places):
        self.__places = new_places

    def set_new_initial_marking(self):
        self.__initial_marking = self.__places.copy()
        self.__initial_marking_sparse = self.__sparse_marking.copy()

    def get_number_of_tokens(self):
        total_tokens = sum(list(self.__sparse_marking.values()))
        return total_tokens

    def get_sparse_marking(self):
        return self.__sparse_marking.copy()

    def get_initial_sparse_marking(self):
        return self.__initial_marking_sparse.copy()

    def get_transition_rate(self, transition):
        tr_info = self.__transitions[transition]
        rate = tr_info[-1]
        return rate

    def get_place_marking(self, place_name):
        ntokens = self.__places.get(place_name)
        return ntokens

    def set_place_marking(self, place_name, ntokens):
        self.__places.update({place_name:ntokens})
        return

    def rename_place(self, place, new_name):
        if place == new_name:
            return False

        ntokens = self.__places[place]
        place_index = self.places_to_index[place]

        if ntokens > 0:
            self.__sparse_marking[new_name] = ntokens
            del self.__sparse_marking[place]

        self.__places[new_name] = ntokens
        del self.__places[place]

        self.places_to_index[new_name] = place_index
        del self.places_to_index[place]

        self.index_to_places[place_index] = new_name

        self.__initial_marking = self.__places.copy()
        self.__initial_marking_sparse = self.__sparse_marking.copy()

        return True

    def add_places(self, name, ntokens=[], set_initial_marking=True):
        '''
        Adds new places to the existing ones in the GSPN object. Replaces the ones with the same name.

        :param name: (list str) denoting the name of the places
        :param ntokens: (list int) denoting the current number of tokens of the given places
        :param set_initial_marking: (bool) denoting whether we want to define ntokens as the initial marking or not
        '''

        lenPlaces = len(self.__places)
        for index, place_name in enumerate(name):
            if ntokens:
                self.__places[place_name] = ntokens[index]
                if ntokens[index] > 0:
                    self.__sparse_marking[place_name] = ntokens[index]
            else:
                self.__places[place_name] = 0

            self.places_to_index[place_name] = lenPlaces
            self.index_to_places[lenPlaces] = place_name
            lenPlaces += 1

        if set_initial_marking:
            self.__initial_marking = self.__places.copy()
            self.__initial_marking_sparse = self.__sparse_marking.copy()

        return self.__places.copy()

    def rename_transition(self, transition, new_name):
        if transition == new_name:
            return False

        tr_info = self.__transitions[transition]
        transition_index = self.transitions_to_index[transition]

        self.__transitions[new_name] = tr_info
        del self.__transitions[transition]

        self.transitions_to_index[new_name] = transition_index
        del self.transitions_to_index[transition]

        self.index_to_transitions[transition_index] = new_name

        self.__imm_transitions_generated = False
        self.__timed_transitions_generated = False

        return True

    def change_transition_rate(self, transition_name, new_rate):
        tr_info = self.__transitions[transition_name]
        tr_info[1] = new_rate
        self.__transitions[transition_name] = tr_info

        self.__imm_transitions_generated = False
        self.__timed_transitions_generated = False

    def add_transitions(self, tname, tclass=[], trate=[]):
        '''
        Adds new transitions to the existing ones in the GSPN object. Replaces the ones with the same name.

        :param tname: (list str) denoting the name of the transition
        :param tclass: (list str) indicating if the corresponding transition is either immediate ('imm') or exponential ('exp')
        :param trate: (list float) representing a static firing rate in an exponential transition and a static (non marking dependent) weight in a immediate transition
        :return: (dict) all the transitions of the GSPN object
        '''

        lenTransitions = len(self.__transitions)
        for index, transition_name in enumerate(tname):
            self.__transitions[transition_name] = []
            if tclass:
                self.__transitions[transition_name].append(tclass[index])
            else:
                raise Exception('No transition type defined for transition '+str(tclass[index]))
            if trate:
                self.__transitions[transition_name].append(trate[index])
            else:
                raise Exception('No transition rate defined for transition '+str(tclass[index]))

            self.transitions_to_index[tname[index]] = lenTransitions
            self.index_to_transitions[lenTransitions] = tname[index]
            lenTransitions += 1

        self.__imm_transitions_generated = False
        self.__timed_transitions_generated = False

        return self.__transitions.copy()

    def replace_arcs_sparse_matrices(self, new_arc_in, new_arc_out):
        self.__arc_in_m = new_arc_in
        self.__arc_out_m = new_arc_out
        return True

    def add_arcs(self, arc_in, arc_out):
        '''
        example:
        arc_in[<output place>] = [(<input transition1>, <arc weight>),
                                  (<input transition2>, <arc weight>), ...]

        arc_out[<output transition>] = [(<input place1>, <arc weight>),
                                        (<input place2>, <arc weight>), ...]

        arc_in = {}
        arc_in['p1'] = [('t1', 2), ('t2', 5)]
        arc_in['p2'] = [('t3', 1)]
        ...

        arc_out = {}
        arc_out['t1'] = [('p2', 1)]
        arc_out['t2'] = [('p5', 1), ('p1', 5)]
        ...

        :param arc_in: (dict) mapping the arc connections from places to transitions
        :param arc_out: (dict) mapping the arc connections from transitions to places
        :return: (sparse COO, sparse COO)
        arc_in_m -> Sparse COO matrix where the x coordinates holds the source place index and the y coordinates
                   the target transition index.
        arc_out_m -> Sparse COO matrix where the x coordinates holds the source transition index and the y coordinates
                   the target place index.
        Each pair place-transition kept in the sparse matrix corresponds to an input/output connecting arc in the
        GSPN.
        '''

        aux_in_list = [[], []]
        aux_in_list[0] = self.__arc_in_m.coords[0].tolist()
        aux_in_list[1] = self.__arc_in_m.coords[1].tolist()
        aux_in_weights = self.__arc_in_m.data.tolist()
        for place_in, list_transitions_in in arc_in.items():
            for arc_transition_in in list_transitions_in:
                transition_in = arc_transition_in[0]
                arc_weight = arc_transition_in[1]
                aux_in_list[0].append(self.places_to_index[place_in])
                aux_in_list[1].append(self.transitions_to_index[transition_in])
                aux_in_weights.append(arc_weight)

        aux_out_list = [[], []]
        aux_out_list[0] = self.__arc_out_m.coords[0].tolist()
        aux_out_list[1] = self.__arc_out_m.coords[1].tolist()
        aux_out_weights = self.__arc_out_m.data.tolist()
        for transition_out, list_places_out in arc_out.items():
            for arc_place_out in list_places_out:
                place_out = arc_place_out[0]
                arc_weight = arc_place_out[1]
                aux_out_list[0].append(self.transitions_to_index[transition_out])
                aux_out_list[1].append(self.places_to_index[place_out])
                aux_out_weights.append(arc_weight)

        #  Creation of Sparse Matrix
        if arc_in:
            self.__arc_in_m = sparse.COO(aux_in_list, aux_in_weights,
                                         shape=(len(self.__places), len(self.__transitions)))
        if arc_out:
            self.__arc_out_m = sparse.COO(aux_out_list, aux_out_weights,
                                          shape=(len(self.__transitions), len(self.__places)))

        return self.__arc_in_m, self.__arc_out_m

    def add_tokens(self, place_name, ntokens, set_initial_marking=False):
        '''
        Adds extra tokens to the places in the place_name list.
        :param place_name: (list) with the input places names, to where the tokens should be added
        :param ntokens: (list) with the number of tokens to be added (must have the same order as in the place_name list)
        :param set_initial_marking: (bool) if True the number of tokens added will also be added to the initial
                                    marking, if False the initial marking remains unchanged
        :return: (bool) True if successful, and False otherwise
        '''
        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                tokens_to_add = ntokens.pop()
                if self.__places[p] != 'w':
                    if tokens_to_add == 'w':
                        self.__places[p] = 'w'
                        self.__sparse_marking[p] = 'w'
                    elif tokens_to_add > 0:
                        self.__places[p] = self.__places[p] + tokens_to_add
                        self.__sparse_marking[p] = self.__places[p]

            if set_initial_marking:
                self.__initial_marking = self.__places.copy()
                self.__initial_marking_sparse = self.__sparse_marking.copy()

            return True
        else:
            return False

    def remove_tokens(self, place_name, ntokens, set_initial_marking=False):
        '''
        Removes tokens from the places in the place_name list.
        :param place_name: (list) with the input places names, from where the tokens should be removed.
        :param ntokens: (list) with the number of tokens to be removed (must have the same order as in the place_name list)
        :param set_initial_marking: (bool) if True the number of tokens removed will also be added to the initial
                                    marking, if False the initial marking remains unchanged.
        :return: (bool) True if successful, and False otherwise
        '''

        if len(place_name) == len(ntokens):
            place_name.reverse()
            ntokens.reverse()
            while place_name:
                p = place_name.pop()
                tokens_to_remove = ntokens.pop()
                if tokens_to_remove == 'w':
                    self.__places[p] = 0
                    del self.__sparse_marking[p]
                elif tokens_to_remove > 0:
                    if self.__places[p] != 'w':
                        self.__places[p] = self.__places[p] - tokens_to_remove
                        self.__sparse_marking[p] = self.__places[p]
                        if self.__sparse_marking[p] <= 0:
                            del self.__sparse_marking[p]

            if set_initial_marking:
                self.__initial_marking = self.__places.copy()
                self.__initial_marking_sparse = self.__sparse_marking.copy()

            return True
        else:
            raise Exception('Number of elements in list of tokens to remove does not match the number of places.')

    def get_places_lists(self):
        n_places = len(self.__places.keys())
        place_list = [0]*n_places
        ntokens_list = [0]*n_places
        for index, (place, ntokens) in enumerate(self.__places.items()):
            place_list[index] = place
            ntokens_list[index] = ntokens

        return place_list, ntokens_list

    def get_transitions_lists(self):
        n_transitions = len(self.__transitions.keys())

        tr_name_list = [0]*n_transitions
        tr_type_list = [0]*n_transitions
        tr_rate_list = [0]*n_transitions
        for index, (tr_name, tr_info) in enumerate(self.__transitions.items()):
            tr_name_list[index] = tr_name
            tr_type_list[index] = tr_info[0]
            tr_rate_list[index] = tr_info[1]

        return tr_name_list, tr_type_list, tr_rate_list

    def merge_gspn(self, gspn):
        # merge places
        place_list, ntokens_list = gspn.get_places_lists()
        self.add_places(place_list, ntokens_list, set_initial_marking=True)

        # merge transitions
        tr_name_list, tr_type_list, tr_rate_list = gspn.get_transitions_lists()
        self.add_transitions(tr_name_list, tr_type_list, tr_rate_list)

        # merge arcs
        arcs_in, arcs_out = gspn.get_arcs_dict()
        self.add_arcs(arcs_in, arcs_out)

        self.__imm_transitions_generated = False
        self.__timed_transitions_generated = False

    def get_current_marking(self, sparse_marking=False):
        if sparse_marking:
            return self.__sparse_marking.copy()
        else:
            return self.__places.copy()
        
    def get_current_marking_list(self):
        marking_list = []
        for place, mark in self.__places.items():
            marking_list.append(mark)
        return marking_list
    
    def set_marking_list(self, mark_list):
        place_list = list(self.__places)
        
        if len(mark_list) == len(place_list):
            self.__sparse_marking = {}
            indexy = 0
            for pl, tk in self.__places.items():
                self.__places.update({pl:mark_list[indexy]})
                if mark_list[indexy] > 0:
                    self.__sparse_marking[pl] = mark_list[indexy]
                indexy += 1
        else:
            print("error: place number and provided marking are of different size!")

    def set_marking(self, places):
        self.__places = places.copy()
        self.__sparse_marking = {}
        for pl, tk in self.__places.items():
            if tk > 0:
                self.__sparse_marking[pl] = tk
        return True

    def get_initial_marking(self, sparse_marking=False):
        if sparse_marking:
            return self.__initial_marking_sparse.copy()
        else:
            return self.__initial_marking.copy()

    def get_transitions(self):
        return self.__transitions.copy()

    def get_imm_transitions(self):
        if not self.__imm_transitions_generated:
            for tr_name, tr_info in self.__transitions.items():
                tr_type = tr_info[0]
                tr_rate = tr_info[1]
                if tr_type == 'imm':
                    self.__imm_transitions[tr_name] = tr_rate

            self.__imm_transitions_generated = True

        return self.__imm_transitions.copy()

    def get_timed_transitions(self):
        if not self.__timed_transitions_generated:
            for tr_name, tr_info in self.__transitions.items():
                tr_type = tr_info[0]
                tr_rate = tr_info[1]
                if tr_type == 'exp':
                    self.__timed_transitions[tr_name] = tr_rate

            self.__timed_transitions_generated = True

        return self.__timed_transitions.copy()

    def get_arcs(self):
        return self.__arc_in_m.copy(), self.__arc_out_m.copy()

    def get_arcs_dict(self):
        '''
        Converts the arcs sparse matrices to dicts and outputs them.
        :return: arcs in dict form
        '''
        arcs_in = {}
        for iterator in range(len(self.__arc_in_m.coords[0])):
            out_place = self.__arc_in_m.coords[0][iterator]
            out_place = self.index_to_places[out_place]
            in_tr = self.__arc_in_m.coords[1][iterator]
            in_tr = self.index_to_transitions[in_tr]
            arc_weight = self.__arc_in_m.data[iterator]
            if out_place in arcs_in:
                arcs_in[out_place].append((in_tr, arc_weight))
            else:
                arcs_in[out_place] = [(in_tr, arc_weight)]

        arcs_out = {}
        for iterator in range(len(self.__arc_out_m.coords[0])):
            out_tr = self.__arc_out_m.coords[0][iterator]
            out_tr = self.index_to_transitions[out_tr]
            in_place = self.__arc_out_m.coords[1][iterator]
            in_place = self.index_to_places[in_place]
            arc_weight = self.__arc_out_m.data[iterator]
            if out_tr in arcs_out:
                arcs_out[out_tr].append((in_place, arc_weight))
            else:
                arcs_out[out_tr] = [(in_place, arc_weight)]

        return arcs_in, arcs_out

    def get_connected_arcs(self, name, type):
        '''
        Returns input and output arcs connected to a given element (place/transition) of the Petri Net
        :param name: (str) Name of the element (place name or transition name)
        :param type: (str) Either the string 'place' or 'transition' to indicate if the input is a place or a transition
        :return: (list, list) Lists of input and output elements connected to the input element
        '''

        if type != 'transition' and type != 'place':
            raise Exception("Argument type not specified. Choose between 'transition' and 'place'.")

        if type == 'place':
            arcs_in_aux, arcs_out_aux = self.get_arcs_dict()

            arcs_in = []
            for transition in arcs_out_aux.keys():
                for arc_info in arcs_out_aux[transition]:
                    place = arc_info[0]
                    if place == name:
                        arc_weight = arc_info[1]
                        arcs_in.append((transition, arc_weight))

            arcs_out = arcs_in_aux[name]

        if type == 'transition':
            arcs_in_aux, arcs_out_aux = self.get_arcs_dict()

            arcs_in = []
            for place in arcs_in_aux.keys():
                for arc_info in arcs_in_aux[place]:
                    transition = arc_info[0]
                    if transition == name:
                        arc_weight = arc_info[1]
                        arcs_in.append((place, arc_weight))

            arcs_out = arcs_out_aux[name]

        return arcs_in, arcs_out

    def remove_place(self, place):
        '''
        Method that removes PLACE from Petri Net, with corresponding connected input and output arcs
        :param (str) Name of the place to be removed
        :return: (dict)(dict) Dictionaries containing input and output arcs connected to the removed place
        '''
        place_id = self.places_to_index[place]

        # remove place from places tracker
        self.__places.pop(place)
        self.places_to_index.pop(place)
        self.index_to_places.pop(place_id)
        if place in self.__sparse_marking:
            self.__sparse_marking.pop(place)

        # renormalize places index
        largest_idx = 0
        for pl, pl_idx in self.places_to_index.copy().items():
            if pl_idx > largest_idx:
                largest_idx = int(pl_idx)
            if pl_idx > place_id:
                self.places_to_index[pl] -= 1
                self.index_to_places[pl_idx-1] = self.index_to_places[pl_idx]
        # delete largest index, as we decreased all the indexes and this no longer corresponds to any placee
        self.index_to_places.pop(largest_idx)

        # removing place from arc_in and decrease place index
        places_list = self.__arc_in_m.coords[0].tolist()
        transitions_list = self.__arc_in_m.coords[1].tolist()
        arc_weight_list = self.__arc_in_m.data.tolist()
        for i in reversed(range(len(places_list))):
            if places_list[i] == place_id:
                del places_list[i]
                del transitions_list[i]
                del arc_weight_list[i]
            elif places_list[i] > place_id:
                places_list[i] -= 1
        # create new sparse matrix for arc_in
        self.__arc_in_m = sparse.COO([places_list, transitions_list], arc_weight_list,
                                     shape=(len(self.__places), len(self.__transitions)))

        # remove place from arc_out and decrease place index
        transitions_list = self.__arc_out_m.coords[0].tolist()
        places_list = self.__arc_out_m.coords[1].tolist()
        arc_weight_list = self.__arc_out_m.data.tolist()
        for i in reversed(range(len(places_list))):
            if places_list[i] == place_id:
                del places_list[i]
                del transitions_list[i]
                del arc_weight_list[i]
            elif places_list[i] > place_id:
                places_list[i] -= 1
        # create new sparse matrix for arc_out
        self.__arc_out_m = sparse.COO([transitions_list, places_list], arc_weight_list,
                                      shape=(len(self.__transitions), len(self.__places)))

        return True

    def remove_transition(self, transition):
        '''
        Method that removes TRANSITION from Petri Net, with corresponding input and output arcs
        :param transition:(str) Name of the transition to be removed
        :return: (bool) True
        '''
        transition_id = self.transitions_to_index[transition]

        # remove transition from transitions tracker
        self.__transitions.pop(transition)
        self.transitions_to_index.pop(transition)
        self.index_to_transitions.pop(transition_id)

        # renormalize transition index
        largest_idx = 0
        for tr, tr_idx in self.transitions_to_index.copy().items():
            if tr_idx > largest_idx:
                largest_idx = int(tr_idx)
            if tr_idx > transition_id:
                self.transitions_to_index[tr] -= 1
                self.index_to_transitions[tr_idx-1] = self.index_to_transitions[tr_idx]
        # delete largest index, as we decreased all the indexes and this no longer corresponds to any transition
        self.index_to_transitions.pop(largest_idx)

        # removing transition from arc_in
        places_list = self.__arc_in_m.coords[0].tolist()
        transitions_list = self.__arc_in_m.coords[1].tolist()
        arc_weight_list = self.__arc_in_m.data.tolist()
        for i in reversed(range(len(transitions_list))):
            if transitions_list[i] == transition_id:
                del transitions_list[i]
                del places_list[i]
                del arc_weight_list[i]
            elif transitions_list[i] > transition_id:
                transitions_list[i] -= 1

        # creating new sparse for arc_in
        self.__arc_in_m = sparse.COO([places_list, transitions_list], arc_weight_list,
                                     shape=(len(self.__places), len(self.__transitions)))

        # removing transition from arc_out
        transitions_list = self.__arc_out_m.coords[0].tolist()
        places_list = self.__arc_out_m.coords[1].tolist()
        arc_weight_list = self.__arc_out_m.data.tolist()
        for i in reversed(range(len(transitions_list))):
            if transitions_list[i] == transition_id:
                del transitions_list[i]
                del places_list[i]
                del arc_weight_list[i]
            elif transitions_list[i] > transition_id:
                transitions_list[i] -= 1

        # creating new sparse for arc_out
        self.__arc_out_m = sparse.COO([transitions_list, places_list], arc_weight_list,
                                      shape=(len(self.__transitions), len(self.__places)))

        self.__imm_transitions_generated = False
        self.__timed_transitions_generated = False

        return True

    def remove_arc(self, arcs_in=None, arcs_out=None):
        '''
        Method that removes ARCS from Petri Net.
        :param arcs_in: (dict) Dictionary containing all input arcs to be deleted: e.g.  arcs_in[p1]=['t1','t2'], arcs_in[p2]=['t1','t3']
        :param arcs_out: (dict) Dictionary containing output arcs to be deleted: e.g. arcs_out[t1]=['p1','p2'], arcs_out[t2]=['p1','p3']
        :return: Boolean
        '''

        if arcs_in == None and arcs_out == None:
            return False

        if arcs_in != None:
            for place in arcs_in.keys():
                place_id = self.places_to_index[place]  # This is an int
                for transition in arcs_in[place]:
                    transition_id = self.transitions_to_index[transition]  # This is an int
                    places_list = self.__arc_in_m.coords[0].tolist()
                    transitions_list = self.__arc_in_m.coords[1].tolist()
                    arc_index = self.find_index_value(places_list, transitions_list, place_id, transition_id)
                    if arc_index != None:
                        arc_weights_list = self.__arc_in_m.data.tolist()
                        del places_list[arc_index]
                        del transitions_list[arc_index]
                        del arc_weights_list[arc_index]
                        self.__arc_in_m = sparse.COO([places_list, transitions_list], arc_weights_list)

        if arcs_out != None:
            for transition in arcs_out.keys():
                transition_id = self.transitions_to_index[transition]  # This is an int
                for place in arcs_out[transition]:
                    place_id = self.places_to_index[place]  # This is an int
                    transitions_list = self.__arc_out_m.coords[0].tolist()
                    places_list = self.__arc_out_m.coords[1].tolist()
                    arc_index = self.find_index_value(transitions_list, places_list, transition_id, place_id)
                    if arc_index != None:
                        arc_weights_list = self.__arc_out_m.data.tolist()
                        del places_list[arc_index]
                        del transitions_list[arc_index]
                        del arc_weights_list[arc_index]
                        self.__arc_out_m = sparse.COO([transitions_list, places_list], arc_weights_list)
        return True
    
    def create_place_to_transition_arc(self, arc_in, arc_out, trans_name, place_name, weight=None, P2T_mode=True):
        '''
            Fills out arc data between a place and transition 
        '''
        if weight == None:
            arcWeight = 1
        else:
            arcWeight = weight

        if P2T_mode:
            if (arc_in.get(place_name) != None):
                arc_in[place_name].append([trans_name, arcWeight])
            else:
                arc_in[place_name] = [[trans_name, arcWeight]]
        else:
            if (arc_out.get(trans_name) != None):
                arc_out[trans_name].append([place_name, arcWeight])
            else:
                arc_out[trans_name] = [[place_name, arcWeight]]
        return


    def create_enabling_arc(self, arc_in, arc_out, trans_name, place_name):
        '''
            Fills out enabling arc data between a place and transition 
        '''
        self.place_to_transition_arc(arc_in, arc_out, trans_name, place_name, None, True)
        self.place_to_transition_arc(arc_in, arc_out, trans_name, place_name, None, False)
        return

    def find_index_value(self, list1, list2, element1, element2):
        '''
        Return the first index where there is overlap between list1 and list2 elements.
        :param list1: list of int
        :param list2: list of int
        :param element1: element we want to find on list 1
        :param element2: element we want to find on list 2
        :return: index of the elements
        '''
        for i in list1:
            if list1[i] == element1:
                if list2[i] == element2:
                    return i

        return None


    def prep_arc_data(self):
        input_places = {}
        output_places = {}
        for list_index, transition in enumerate(self.__arc_in_m.coords[1]):
            place = self.__arc_in_m.coords[0][list_index]
            arc_weight = self.__arc_in_m.data[list_index]
            if transition in input_places.keys():
                input_places[transition].append((place, arc_weight))
            else:
                input_places[transition] = [(place, arc_weight)]

        for list_index, transition in enumerate(self.__arc_out_m.coords[0]):
            place = self.__arc_out_m.coords[1][list_index]
            #place = self.index_to_places[place]
            arc_weight = self.__arc_out_m.data[list_index]
            if transition in output_places.keys():
                output_places[transition].append((place, arc_weight))
            else:
                output_places[transition] = [(place, arc_weight)]
        self.input_places = input_places
        self.output_places = output_places


    def get_enabled_transitions(self, place_name=None, w_prep=False):
        """
        param place_name: (str) place name. When this arg is passed the method returns only the
                           enabled transitions connected to that place
        :return: (dict) where the keys hold the enabled transitions id and the values
                        the rate/weight of each transition
        """

        enabled_exp_transitions = {}
        random_switch = {}
        current_marking = self.__places.copy()
        # dict where the keys are transitions and the values the corresponding input places
        input_places = {}

        if place_name:
            place_index = self.places_to_index[place_name]

            output_transitions = []
            # for each transition get all the places that have an input arc connection
            for list_index, transition in enumerate(self.__arc_in_m.coords[1]):
                place = self.__arc_in_m.coords[0][list_index]
                arc_weight = self.__arc_in_m.data[list_index]

                if place == place_index:
                    output_transitions.append(transition)

                if transition in input_places.keys():
                    input_places[transition].append((place, arc_weight))
                else:
                    input_places[transition] = [(place, arc_weight)]

            # for all transitions check the ones that are enabled
            # i.e. the input places have at least same number of tokens as the input arc weight
            for tr, list_places in input_places.items():
                enabled_transition = True
                for in_pl, in_weight in list_places:
                    in_pl_name = self.index_to_places[in_pl]
                    # if current_marking[in_pl_name] == 0:
                    if current_marking[in_pl_name] < in_weight:
                        enabled_transition = False
                        break

                if enabled_transition and tr in output_transitions:
                    tr_name = self.index_to_transitions[tr]
                    if self.__transitions[tr_name][0] == 'exp':
                        enabled_exp_transitions[tr_name] = self.__transitions[tr_name][1]
                    else:
                        random_switch[tr_name] = self.__transitions[tr_name][1]
        else:
            if not(w_prep):
                # for each transition get all the places that have an input arc connection
                for list_index, transition in enumerate(self.__arc_in_m.coords[1]):
                    place = self.__arc_in_m.coords[0][list_index]
                    arc_weight = self.__arc_in_m.data[list_index]
                    if transition in input_places.keys():
                        input_places[transition].append((place, arc_weight))
                    else:
                        input_places[transition] = [(place, arc_weight)]
            else:
                input_places = self.input_places

            # for all transitions check the ones that are enabled
            # i.e. the input places have at least same number of tokens as the input arc weight
            for tr, list_places in input_places.items():
                enabled_transition = True
                for in_pl, in_weight in list_places:
                    in_pl_name = self.index_to_places[in_pl]
                    # if current_marking[in_pl_name] == 0:
                    if current_marking[in_pl_name] < in_weight:
                        enabled_transition = False
                        break

                if enabled_transition:
                    tr_name = self.index_to_transitions[tr]
                    if self.__transitions[tr_name][0] == 'exp':
                        enabled_exp_transitions[tr_name] = self.__transitions[tr_name][1]
                    else:
                        random_switch[tr_name] = self.__transitions[tr_name][1]

        return enabled_exp_transitions.copy(), random_switch.copy()

    def fire_transition(self, transition, w_prep=False):
        '''
        Removes 1 token from all input places and adds 1 token to all the output places of the given transition.
        :param transition: (str) name of the transition to be fired.
        :return: always returns True
        '''
        input_places = []
        output_places = []
        input_tokens = []
        output_tokens = []
        transition_index = self.transitions_to_index[transition]

        if not(w_prep):
            # get a list with all the input places of the given transition
            for list_index, transition in enumerate(self.__arc_in_m.coords[1]):
                if transition == transition_index:
                    place = self.__arc_in_m.coords[0][list_index]
                    place = self.index_to_places[place]
                    tokens = self.__arc_in_m.data[list_index]
                    input_places.append(place)
                    input_tokens.append(tokens)
        else:
            data = self.input_places.get(transition_index)
            for data_element in data:
                place_name = self.index_to_places[data_element[0]]
                input_places.append(place_name)
                input_tokens.append(data_element[1])

        if not(w_prep):
            # get a list with all the output places of the given transition
            for list_index, transition in enumerate(self.__arc_out_m.coords[0]):
                if transition == transition_index:
                    place = self.__arc_out_m.coords[1][list_index]
                    place = self.index_to_places[place]
                    tokens = self.__arc_out_m.data[list_index]
                    output_places.append(place)
                    output_tokens.append(tokens)
        else:
            data = self.output_places.get(transition_index)
            for data_element in data:
                place_name = self.index_to_places[data_element[0]]
                output_places.append(place_name)
                output_tokens.append(data_element[1])

        # remove tokens from input places
        self.remove_tokens(input_places, input_tokens)
        # add tokens to output places
        self.add_tokens(output_places, output_tokens)

        return True

    def simulate(self, nsteps=1, reporting_step=1, simulate_wait=False):
        markings = []
        fired_transitions = []
        fired_transition = 0
        for step in range(nsteps):
            if step % reporting_step == 0:
                markings.append(self.get_current_marking())

            enabled_exp_transitions, random_switch = self.get_enabled_transitions()

            if random_switch:
                if len(random_switch) > 1:
                    s = sum(random_switch.values())
                    random_switch_id = []
                    random_switch_prob = []
                    # normalize the associated probabilities
                    for key, value in random_switch.items():
                        random_switch_id.append(key)
                        random_switch_prob.append(value / s)

                    # Draw from all enabled immediate transitions
                    firing_transition = np.random.choice(a=random_switch_id, size=None, p=random_switch_prob)
                    # Fire transition
                    fired_transition = firing_transition
                    self.fire_transition(firing_transition)
                    fired_transitions.append(firing_transition)
                else:
                    # Fire the only available immediate transition
                    fired_transition = list(random_switch.keys())[0]
                    self.fire_transition(fired_transition)
                    fired_transitions.append(fired_transition)
            elif enabled_exp_transitions:
                if len(enabled_exp_transitions) > 1:
                    if simulate_wait:
                        wait_times = enabled_exp_transitions.copy()
                        # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
                        # in this case the beta rate parameter is used instead, where beta = 1/lambda
                        for key, value in enabled_exp_transitions.items():
                            wait_times[key] = np.random.exponential(scale=(1.0 / value), size=None)

                        firing_transition = min(wait_times, key=wait_times.get)
                        wait = wait_times[firing_transition]
                        time.sleep(wait)

                    else:
                        s = sum(enabled_exp_transitions.values())
                        exp_trans_id = []
                        exp_trans_prob = []
                        # normalize the associated probabilities
                        for key, value in enabled_exp_transitions.items():
                            exp_trans_id.append(key)
                            exp_trans_prob.append(value / s)

                        # Draw from all enabled exponential transitions
                        firing_transition = np.random.choice(a=exp_trans_id, size=None, p=exp_trans_prob)

                    # Fire transition
                    fired_transition = firing_transition
                    self.fire_transition(firing_transition)
                    fired_transitions.append(firing_transition)
                else:
                    if simulate_wait:
                        wait = np.random.exponential(scale=(1.0 / list(enabled_exp_transitions.values())[0]), size=None)
                        time.sleep(wait)

                    # Fire transition
                    fired_transition = list(enabled_exp_transitions.keys())[0]
                    self.fire_transition(fired_transition)
                    fired_transitions.append(fired_transition)

        full_list = list(markings)
        full_list.append(self.get_current_marking())
        full_list.append(fired_transitions)
        return full_list

    def get_state_from_marking(self, marking, states_to_marking):
        for st, mk in states_to_marking.items():
            if mk == marking:
                return st

        return None

    def simulate_policy(self, policy, states_to_marking, partial_policy=True, simulate_wait=False):
        enabled_exp_transitions, enabled_imm_transitions = self.get_enabled_transitions()

        current_state = self.get_state_from_marking(marking=self.__sparse_marking, states_to_marking=states_to_marking)
        if current_state in policy:
            action = policy[current_state]
        else:
            action = None

        if action == None:
            if partial_policy:
                if enabled_imm_transitions:
                    weight_sum = sum(enabled_imm_transitions.values())
                    if weight_sum == 0:
                        firing_transition = np.random.choice(a=list(enabled_imm_transitions.keys()))
                    else:
                        random_switch_id = []
                        random_switch_prob = []
                        # normalize the associated probabilities
                        for key, value in enabled_imm_transitions.items():
                            random_switch_id.append(key)
                            random_switch_prob.append(value / weight_sum)

                        # Draw from all enabled immediate transitions
                        firing_transition = np.random.choice(a=random_switch_id, size=None, p=random_switch_prob)

                    wait_until_fire = 0
                    # Fire transition
                    self.fire_transition(firing_transition)
                elif enabled_exp_transitions:
                    wait_times = enabled_exp_transitions.copy()
                    # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
                    # in this case the beta rate parameter is used instead, where beta = 1/lambda
                    for key, value in enabled_exp_transitions.items():
                        wait_times[key] = np.random.exponential(scale=(1.0 / value), size=None)

                    firing_transition = min(wait_times, key=wait_times.get)
                    wait_until_fire = wait_times[firing_transition]
                    if simulate_wait:
                        time.sleep(wait_until_fire)

                    # Fire transition
                    self.fire_transition(firing_transition)
                else:
                    raise Exception('Deadlock, there are no enabled transitions in marking: ' + str(self.__sparse_marking))
            else:
                raise Exception('Policy not defined for marking: ' + str(self.__sparse_marking))
        elif action in enabled_imm_transitions:
            firing_transition = action
            wait_until_fire = 0
            self.fire_transition(firing_transition)
        elif action in ['EXP', 'WAIT']:
            if enabled_exp_transitions:
                wait_times = enabled_exp_transitions.copy()
                # sample from each exponential distribution prob_dist(x) = lambda * exp(-lambda * x)
                # in this case the beta rate parameter is used instead, where beta = 1/lambda
                for key, value in enabled_exp_transitions.items():
                    wait_times[key] = np.random.exponential(scale=(1.0 / value), size=None)

                firing_transition = min(wait_times, key=wait_times.get)
                wait_until_fire = wait_times[firing_transition]
                if simulate_wait:
                    time.sleep(wait_until_fire)

                # Fire transition
                self.fire_transition(firing_transition)
            else:
                raise Exception('Action: '+str(action)+' does not match with any of the enabled exp transitions '
                                'in the marking: '+str(self.__sparse_marking))
        else:
            raise Exception('Action: '+str(action)+' does not match with any enabled transition in the marking: '
                            +str(self.__sparse_marking))

        # print('Fired transition : ', firing_transition)
        return firing_transition, wait_until_fire, self.get_current_marking(sparse_marking=False), self.get_current_marking(sparse_marking=True)

    def reset_simulation(self):
        self.__places = self.__initial_marking.copy()
        self.__sparse_marking = self.__initial_marking_sparse.copy()
        return True

    def reset(self):
        self.__places = self.__initial_marking.copy()
        self.__sparse_marking = self.__initial_marking_sparse.copy()
        self.__nsamples = {}
        self.__sum_samples = {}
        return True

    # def prob_having_n_tokens(self, place_id, ntokens):
    #     '''
    #     Computes the probability of having exactly k tokens in a place pi.
    #     :param place_id: identifier of the place for which the probability
    #     :param ntokens: number of tokens
    #     :return:
    #     '''

    def init_analysis(self):
        self.__ct_tree = gspn_analysis.CoverabilityTree(self)
        self.__ct_tree.generate()
        self.__ctmc = gspn_analysis.CTMC(self.__ct_tree)
        self.__ctmc.generate()
        self.__ctmc.compute_transition_rate()
        self.__ctmc_steady_state = self.__ctmc.get_steady_state()

        self.__ct_ctmc_generated = True

        return True

    def liveness(self):
        '''
        Checks the liveness of a GSPN. If the GSPN is live means that is deadlock free and therefore is able
        to fire some transition no matter what marking has been reached.
        :return: (bool) True if is deadlock free and False otherwise.
        '''
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        return self.__ct_tree.deadlock_free

    def transition_throughput_rate(self, transition):
        '''
        The throughput of an exponential transition tj is computed by considering its firing rate over the probability
        of all states where tj is enabled. The throughput of an immediate transition tj can be computed by considering
        the throughput of all exponential transitions which lead immediately to the firing of transition tj, i.e.,
        without crossing any tangible state, together with the probability of firing transition tj among all the
        enabled immediate transitions.
        :param transition: (string) with the transition id for which the throughput rate will be computed
        :return: (float) with the computed throughput rate
        '''

        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        if self.__transitions[transition][0] == 'exp':
            transition_rate = self.__transitions[transition]
            transition_rate = transition_rate[1]
            states_already_considered = []
            throughput_rate = 0
            for tr in self.__ctmc.transition:
                state = tr[0]
                transiton_id = tr[2]
                transiton_id = transiton_id.replace('/', ':')
                transiton_id = transiton_id.split(':')
                if (transition in transiton_id) and not (state in states_already_considered):
                    throughput_rate = throughput_rate + self.__ctmc_steady_state.loc[state] * transition_rate

                    states_already_considered.append(state)
        else:
            throughput_rate = 0
            states_already_considered = []
            for tr in self.__ctmc.transition:
                add_state = False
                tangible_init_state = tr[0]

                transitons_id = tr[2]
                transition_id_set = transitons_id.split('/')
                for tr in transition_id_set:

                    # check if transition exists in the current transition
                    exists_transition = False
                    transitioning_list = tr.split(':')
                    for trn in transitioning_list:
                        if transition == trn:
                            exists_transition = True
                            add_state = True
                            break

                    # if the given transition is part of this ctmc edge, multiply the throughput rate of the exponential transition by the prob of immediate transition
                    if exists_transition and not (tangible_init_state in states_already_considered):
                        exp_transition = transitioning_list[0]
                        current_state = tangible_init_state
                        for trans in transitioning_list:
                            current_transition = trans
                            for edge in self.__ct_tree.edges:
                                if (edge[0] == current_state) and (edge[2] == current_transition):
                                    current_state = edge[1]
                                    break

                            if current_transition == transition:
                                transition_prob = edge[3]
                                exp_transition_rate = self.__transitions[exp_transition]
                                exp_transition_rate = exp_transition_rate[1]

                                throughput_rate = throughput_rate + self.__ctmc_steady_state.loc[
                                    tangible_init_state] * exp_transition_rate * transition_prob

                if add_state:
                    states_already_considered.append(tangible_init_state)
        return throughput_rate

    def prob_of_n_tokens(self, place, ntokens):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        prob_of_n_tokens = 0
        for state_id, marking in self.__ctmc.state.items():
            marking = marking[0]
            for pl in marking:
                if (place == pl[0]) and (ntokens == pl[1]):
                    prob_of_n_tokens = prob_of_n_tokens + self.__ctmc_steady_state.loc[state_id]

        return prob_of_n_tokens

    def expected_number_of_tokens(self, place):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        # compute the maximum possible number of tokens in the inputed place
        maximum_n_tokens = 0
        for state_id, marking in self.__ctmc.state.items():
            marking = marking[0]
            for pl in marking:
                if (pl[0] == place) and (pl[1] > maximum_n_tokens):
                    maximum_n_tokens = pl[1]

        # sum all the probabilities of having exactly n tokens in the given place
        expected_number_of_tokens = 0
        for ntokens in range(maximum_n_tokens):
            expected_number_of_tokens = expected_number_of_tokens + (ntokens + 1) * self.prob_of_n_tokens(place,
                                                                                                          ntokens + 1)

        return expected_number_of_tokens

    def transition_probability_evolution(self, period, step, initial_states_prob, state):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        time_series = np.arange(0, period, step)
        prob_evo = np.zeros(len(time_series))

        for i, time_interval in enumerate(time_series):
            prob_all_states = self.__ctmc.get_prob_reach_states(initial_states_prob, time_interval)
            prob_evo[i] = prob_all_states.loc[state]

        return prob_evo.copy()

    # def mean_wait_time(self, place):
    #     if not self.__ct_ctmc_generated:
    #         raise Exception(
    #             'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')
    #
    #     in_tr_m, out_tr_m = self.get_arcs()
    #     place_column = out_tr_m[0].index(place)
    #     out_tr_m = np.array(out_tr_m)
    #
    #     set_output_transitions = []
    #     for index in range(1,len(out_tr_m)):
    #         if int(out_tr_m[index,place_column]) > 0:
    #             set_output_transitions.append(out_tr_m[index,0])
    #
    #     sum = 0
    #     for transition in set_output_transitions:
    #         print(transition, self.transition_throughput_rate(transition))
    #         sum = sum + self.transition_throughput_rate(transition)
    #
    #     print(self.expected_number_of_tokens(place) / sum)

    def mean_wait_time(self, place):
        if not self.__ct_ctmc_generated:
            raise Exception(
                'Analysis must be initialized before this method can be used, please use init_analysis() method for that purpose.')

        in_tr_m, _ = self.get_arcs()

        # true/false list stating if there is an input connection or not between the given place and any transition
        idd = self.__arc_in_m.loc[place][:].values > 0
        # list with all the output transitions of the given place
        set_output_transitions = list(self.__arc_in_m.columns[idd].values)

        sum = 0
        for transition in set_output_transitions:
            sum = sum + self.transition_throughput_rate(transition)

        return self.expected_number_of_tokens(place) / sum

    def maximum_likelihood_transition(self, transiton, sample):
        '''
        Use maximum likelihood to iteratively estimate the lambda parameter of the exponential distribution that models the inputed transition
        :param transiton: (string) id of the transition that will be updated
        :param sample: (float) sample obtained from a exponential distribution
        :return: (float) the estimated lambda parameter
        '''
        self.__nsamples[transiton] = self.__nsamples + 1
        self.__sum_samples[transiton] = self.__sum_samples[transiton] + sample
        lb = self.__nsamples[transiton] / self.__sum_samples[transiton]

        tr_info = self.__transitions[transiton]
        tr_info[1] = lb
        self.__transitions[transiton] = tr_info

        return lb


# if __name__ == "__main__":
#     # create a generalized stochastic petri net structure
#     my_pn = GSPN()

#     places = my_pn.add_places(['p1', 'p2', 'p3', 'p4', 'p5'], [1, 0, 1, 0, 1])
#     trans = my_pn.add_transitions(['t1', 't2', 't3', 't4'], ['exp', 'exp', 'exp', 'exp'], [1, 1, 0.5, 0.5])
#     arc_in = {}
#     arc_in['p1'] = ['t1']
#     arc_in['p2'] = ['t2']
#     arc_in['p3'] = ['t3']
#     arc_in['p4'] = ['t4']
#     arc_in['p5'] = ['t1', 't3']
#     arc_out = {}
#     arc_out['t1'] = ['p2']
#     arc_out['t2'] = ['p5', 'p1']
#     arc_out['t3'] = ['p4']
#     arc_out['t4'] = ['p3', 'p5']
#     a, b = my_pn.add_arcs(arc_in, arc_out)

#     places = my_pn.add_places(['p1', 'p2'], [1,1])
#     trans = my_pn.add_transitions(['t1'], ['exp'], [1])
#     arc_in = {}
#     arc_in['p1'] = ['t1']
#     arc_out = {}
#     arc_out['t1'] = ['p2']
#     a, b = my_pn.add_arcs(arc_in, arc_out)
