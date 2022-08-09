#!/usr/bin/env python
import re
import sys, getopt

def parse_aut(file_aut,state_variables,action_variables):
    """
    Reads in an aut file and stores the data in a more workable form.

    INPUTS
    ------
    file:
        .aut file
    state_variables:
        list of which variables are environmental variables
    action_variables:
        list of which variables are input variables

    OUTPUTS
    -------
    state_def:
        dict of states and true variables
    next_states:
        dict of states and valid action and what states come next
    rank:
        dict of rank of the states
    """
    state_def = {}
    next_states = {}
    rank = {}
    fid = open(file_aut,'r')
    lines = fid.readlines()

    # Finds out which variables are true in each state and which actions can be taken from each state
    state = 0
    for (idx,line) in enumerate(lines):
        if (idx%2) == 0:
            state_m = re.findall('State\s(\d+)',line)
            state = state_m[0]
            variables_true = get_true_variables(line,state_variables)
            action_true = get_true_variables(line,action_variables)
            rank[state] = get_rank(line)
            if not action_true:
                action_true = [' ']
            state_def[state] = variables_true
            next_states[state] = action_true
        else:
            successors = get_successors(line)
            next_states[state].append(successors)

    return state_def,next_states,rank

def parse_spec(file_spec):
    """
    Reads in a spec file and extracts the system and environment variables.

    INPUTS
    ------
    file_spec:
        .structuredslugs file

    OUTPUTS
    -------
    env_variables:
        list of environment variables
    sys_variables:
        list of system variables

    """
    sys_variables = []
    env_variables = []
    fid = open(file_spec,'r')
    lines = fid.readlines()

    line_input = lines.index('[INPUT]\n')
    line_output = lines.index('[OUTPUT]\n')
    line_env_init = lines.index('[ENV_INIT]\n')

    for idx in range(line_input+1,line_output):
        state = re.findall('(.*)\n',lines[idx])
        if state[0]:
            env_variables.append(state[0])

    for idx in range(line_output+1,line_env_init):
        act = re.findall('(.*)\n',lines[idx])
        if act[0]:
            sys_variables.append(act[0])
    return env_variables,sys_variables


def get_state_variables(line):
    """
    Gets the state variables from the first line of the automata.

    INPUT
    -----
    line:
        1st line (or any state def line) of aut file

    OUTPUT
    ------
    state_variables:
        list of state variables
    """
    p = re.compile('<|,\s(.*?):')
    m = p.findall(line)
    if m[0] == '':
        m = m[1:]

    return m

def get_rank(line):
    """
    Gets the rank of the state.
    """
    p = re.compile('rank\s(.*?)\s->')
    m = p.findall(line)
    if m[0][0] == '(':
        m = m[0][1:-1]
    else:
        m = m[0]
    return m

def get_true_variables(line,state_variables):
    """
    Gets the variables that are true.
    """
    states = '|'.join(state_variables)
    pre_p = '(' + states + ')' + ':1'
    p = re.compile(pre_p)
    m = p.findall(line)
    return m

def get_successors(line):
    """
    Gets the successors to a state.
    """
    p = re.compile('\s(\d+)')
    m = p.findall(line)
    return m


def update_state(arg_intermediate_states, arg_state_number, arg_skill_to_run, arg_state_def, arg_next_states):
    """ Returns -1, "" if the sequence of states violates the strategy
    """
    state_number_out = arg_state_number
    print("updating intermediate state from state: {}".format(state_number_out))
    for ii in range(len(arg_intermediate_states)-1):
        print("updating intermediate state: previous state: {}, skill we ran: {}, symbols that became true: {}".format(state_number_out, arg_skill_to_run, arg_intermediate_states[ii+1]))
        state_number_out_tmp = find_state_number(arg_state_def, arg_next_states, state_number_out, arg_skill_to_run, arg_intermediate_states[ii+1])
        if state_number_out_tmp == -1:
            return -1, ""
        if arg_next_states[state_number_out_tmp][0] == " " or arg_next_states[state_number_out_tmp][0] == arg_skill_to_run:
            state_number_out = state_number_out_tmp
            previous_skill_out = arg_next_states[state_number_out][0]
        else:
            previous_skill_out = arg_skill_to_run

    print("After updating the state we are at: {}".format(state_number_out))
    # previous_skill_out = arg_next_states[state_number_out][0]
    if arg_next_states[state_number_out][0] == arg_skill_to_run:
        print("There is! We assume this is due to the limitations of abstraction and the last state is really visited twice.")
        print("updating intermediate state: previous state: {}, skill we ran: {}, symbols that became true: {}".format(
            state_number_out, arg_skill_to_run, arg_intermediate_states[-1]))
        state_number_out_tmp = find_state_number(arg_state_def, arg_next_states, state_number_out, arg_skill_to_run,
                                             arg_intermediate_states[-1])
        if state_number_out_tmp == -1:
            return -1, ""
        if arg_next_states[state_number_out_tmp][0] == " ":
            state_number_out = state_number_out_tmp
            previous_skill_out = ' '
        else:
            previous_skill_out = arg_skill_to_run
    return state_number_out, previous_skill_out


def get_repeated_states(state_def):
    """
    Find out which states are repeated and remap the numbering to them.
    """
    same_state = {}
    state_list = state_def.items()
    for ii,(state,variables) in enumerate(state_list):
        for (state_comp,variables_comp) in state_list[(ii+1):]:
            if variables == variables_comp and not state_comp in same_state.keys():
                same_state[state_comp] = state

    return same_state

def write_graphviz(file_gviz,state_def,next_states,rank):
    """
    Writes the automata to a graphviz file.
    """
    # Opens the file, write appropriate heading
    # For each state in the definitions, writes the state to the file
    # Writes the next valid transitions
    # For each of these checks if the state is repeated and changes the number
    # if necessary
    fid = open(file_gviz,'w')
    fid.write('digraph G {\n')
    for (state,variables) in state_def.iteritems():
        state_write = 'state' + state
        action,next_state_list = next_states[state]
        for next_state in next_state_list:
            next_state_write = 'state' + next_state
            fid.write("\t %s -> %s [label=\"%s\"];\n"%(state_write,next_state_write,action))

    # Writes the next definition of what the states are
    for (state,variables) in state_def.iteritems():
        state_write = 'state' + str(state)
        variables_write = '\\n'.join(variables)
        variables_write = 'State ' + str(state) + '\\n' + variables_write + '\\n' + 'rank: ' + rank[state]
        fid.write("\t %s [label=\"%s\"];\n"%(state_write,variables_write))

    fid.write('}')
    fid.close()

    # Removes lines that are the same
    fid = open(file_gviz,'r')
    lines = fid.readlines()
    fid.close()

    lines_seen = set()
    fid = open(file_gviz,'w')
    for line in lines:
        if line not in lines_seen:
            fid.write(line)
            lines_seen.add(line)
    fid.close()

def find_state_number(state_def, next_states, previous_state_number, previous_skill, symbols_true):
    """ Finds the next state number based on the current state, the executed
    skill, and the symbols that become true.

    Returns -1 if no valid next state
    """
    valid_next_states_from_previous = next_states[previous_state_number]
    print("Previous skill: {}".format(previous_skill))
    print("Valid next skill from previous: {}".format(valid_next_states_from_previous[0]))
    # print("upcoming assertion: {}".format(previous_skill == valid_next_states_from_previous[0] or previous_skill + 'b' == valid_next_states_from_previous[0]))
    assert previous_skill == valid_next_states_from_previous[0] or previous_skill + 'b' == valid_next_states_from_previous[0], "The previous skill must have been executable"

    # Find which state number we are in based on the symbols true
    state_number = []
    valid_next_state_numbers = valid_next_states_from_previous[1]
    print("valid next states from previous: {}".format(valid_next_state_numbers))
    # print("symbols true that we try to match: {}".format(symbols_true))

    # print("Previous symbols: {}".format(state_def[previous_state_number]))
    # if set(symbols_true) == set(state_def[previous_state_number]):
    #     print("State stayed the same")
    #     return previous_state_number

    for possible_state in valid_next_state_numbers:
        possible_state_symbols_true = state_def[possible_state]
        print("possible state: {} - with symbols true {}".format(possible_state, possible_state_symbols_true))
        if set(possible_state_symbols_true) == set(symbols_true):
            state_number.append(possible_state)

    print("We chose state: {}".format(state_number))
    if len(state_number) == 1:
        return state_number[0]
    else:
        return -1

    # return state_number[0]


def find_skill_to_run(next_states, state_number):
    skill_to_run = next_states[state_number][0]
    return skill_to_run

def find_symbols(state, symbols):
    sym_state = []
    for sym_name, sym in symbols.items():
        tf = sym.in_symbol(state)
        if sym.in_symbol(state):
            sym_state.append(sym_name)

    return sym_state

def find_intermediate_symbols(states, symbols):
    if type(states) == list:
        states = np.asarray(states)
    # print("States: {}".format(states))
    symbols_out = [find_symbols(states[0, :], symbols)]
    # print("State: {}, symbols: {}".format(states[0, :2], symbols_out[0]))
    for state in states:
        # print("Person: {}".format(p))
        tmp_syms = find_symbols(state, symbols)
        if tmp_syms != symbols_out[-1]:
            tmp_syms2 = find_symbols(state, symbols)
            symbols_out.append(tmp_syms2)

    return symbols_out
