import torch
from rdkit import Chem

def breadth_first_sorting(molecule, node_ranking : list, node_init : int=0):
    '''
        Starting from the specified `node_init` in the graph, uses a breadth-first 
        search (BFS) algorithm to find all adjacent nodes, returning an ordered list 
        of these nodes. Prioritizes the nodes based on the input `node_ranking`. 
        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the 
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int) : Index of node to start the BFS from. Default 0.
        Returns:
        -------
            nodes_visited (list) : BFS ordering for nodes in the molecular graph.
            ref_atoms (list): Possible 3 reference atoms for computing internal coordinates.
    '''

    nodes_visited = [node_init]
    last_nodes_visited = [node_init]
    ref_atoms = [ [] for _ in range(molecule.GetNumAtoms()) ]
    
    A = torch.tensor(Chem.GetAdjacencyMatrix(molecule), dtype = torch.int32)

    ref_atoms[node_init] = [-1, -1, -1]


    # loop until all nodes have been visited
    while len(nodes_visited) < molecule.GetNumAtoms():
        neighboring_nodes = []
        
        for node in last_nodes_visited:
            neighbor_nodes = [int(i) for i in torch.nonzero(A[node, :])]
            new_neighbor_nodes = list(
                set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
            )
            node_importance = [node_ranking[neighbor_node] for
                                neighbor_node in new_neighbor_nodes]

            # check all neighboring nodes and sort in order of importance
            while sum(node_importance) != -len(node_importance):
                next_node = node_importance.index(max(node_importance))
                neighboring_nodes.append(new_neighbor_nodes[next_node])
                node_importance[next_node] = -1

            #Finally protect from reaching to the same atom at the same time
            neighboring_nodes = list(set(neighboring_nodes))
        #Generate reference atoms:
        for new_neighbor_node in neighboring_nodes:
            ref = []
            new_ref = new_neighbor_node #Juts for coding convenience

            #Try to construct a chain of three important connected atoms
            while len(ref) < 3:
                neigh_neigh_nodes = [int(i) for i in torch.nonzero(A[new_ref, :])]
                possible_new_ref = list( (set(neigh_neigh_nodes) & set(nodes_visited)) - set(ref) )
                if possible_new_ref == []:
                    break # This is improvable, add a counter with max the number of options to start the chain and try other neighbours. Otherwise just use the very last and legal ones.
                max_rank = max( [node_ranking[i] for i in possible_new_ref] )
                new_ref = node_ranking.index(max_rank)
                ref.append( new_ref )
            if len(ref) == 3:   # If we managed to build a proper chain, use it as reference
                ref_atoms[new_neighbor_node] = ref.copy()

            else:
                if len(nodes_visited) > 2: # If we have enough atoms, just take the last placed and not used.
                    non_used_atoms = list( set(set(nodes_visited) ) - set(ref) )
                    while len(ref) < 3:
                        ref.append(non_used_atoms[len(ref)-3])                         
                    ref_atoms[new_neighbor_node] = ref.copy()
                else: #If we do not have enough visited atoms, just add dummy tokens -1
                    ref = nodes_visited.copy()
                    while len(ref) < 3:
                        ref.append(-1)
                    ref_atoms[new_neighbor_node] = ref.copy()

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.append( new_neighbor_node )
        # update the list of most recently visited nodes
        last_nodes_visited = set(neighboring_nodes.copy())

    return nodes_visited, ref_atoms