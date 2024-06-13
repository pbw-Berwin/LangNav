''' Batched Room-to-Room navigation environment '''

from hashlib import new
import sys
# sys.path.append('buildpy36')
# sys.path.append('Matterport_Simulator/build/')
# import MatterSim
import csv
import math
import json
import random
import networkx as nx
from utils import angle_feature, load_datasets, load_nav_graphs, pad_instr_tokens, load_cvdn_datasets

csv.field_size_limit(sys.maxsize)

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints'''

    def __init__(self, adj_loc_dict=None, G_list=None):
        self.G_list = G_list
        self.adj_loc_dict = adj_loc_dict

        self.sims = []
        self.connectivity = []

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, viewIds=None):
        if viewIds is None:
            viewIds = [0]*len(scanIds)
        for i, (scanId, viewpointId, viewId) in enumerate(zip(scanIds, viewpointIds, viewIds)):
            if i > len(self.sims) - 1:
                self.sims.append((scanId, viewpointId, viewId))
                self.connectivity.append(self.G_list[scanId])
            else:
                self.sims[i] = (scanId, viewpointId, viewId)
                self.connectivity[i] = self.G_list[scanId]

    def getStates(self):
        states = []
        for i, sim in enumerate(self.sims):
            scanId, viewpointId, viewId = sim      
            connectivity_graph = self.connectivity[i]
            long_id = self._make_id(scanId, viewpointId)
            candidate_adj_loc = self.adj_loc_dict[long_id]
            neighbor_ids = [n for n in connectivity_graph.neighbors(viewpointId)]
            assert len(neighbor_ids) == len(candidate_adj_loc), "The number of neighbors is not equal to the number of candidate adjacent locations"

            state = {
                'scanId' : scanId,
                'viewpoint' : viewpointId,
                'viewIndex' : viewId,
                'adj_loc' : candidate_adj_loc,
            }
            states.append(state)

        return states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be a (viewId) tuple. '''
        for i, action in enumerate(actions):
            if action == -1:
                continue
            scanId, viewpointId, viewId = self.sims[i]
            long_id = self._make_id(scanId, viewpointId)
            candidate_adj_loc = self.adj_loc_dict[long_id]
            candidate_viewpoints = [loc['viewpointId'] for loc in candidate_adj_loc]
            candidate_viewIds = [loc['pointId'] for loc in candidate_adj_loc]
            assert action in candidate_viewIds, "The action is not in the candidate adjacent locations"

            index = candidate_viewIds.index(action)
            next_viewpointId = candidate_viewpoints[index]
            self.sims[i] = (scanId, next_viewpointId, action)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pre-extracted captions '''
    def __init__(self, batch_size=1, seed=10, splits=['train'], task='r2r', ndh_history=None, path_type=None, traj_file=None):
        # self.feat_dict = feature_dict

        self.data = []
        scans = []
        if task == 'r2r':
            selected_path_ids = None
            self.selected_trajs = None
            if traj_file is not None:
                with open(traj_file, 'r') as f:
                    trajs = json.load(f)
                    selected_path_ids = list(trajs.keys())
                    self.selected_trajs = trajs

            for split in splits:
                for i_item, item in enumerate(load_datasets([split])):
                    # !!! Only for evaluation envs
                    # Split multiple instructions into separate entries
                    if '72' in split:
                        new_item = dict(item)
                        new_item['instr_id'] = item['instr_id']
                        new_item['instructions'] = item['instructions']
                        if new_item['instructions'] is not None:  # Filter the wrong data
                            self.data.append(new_item)
                            scans.append(item['scan'])
                    else:
                        for j, instr in enumerate(item['instructions']):
                            new_item = dict(item)
                            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                            if selected_path_ids is not None:
                                if new_item['instr_id'] not in selected_path_ids:
                                    continue
                            new_item['instructions'] = instr
                            if new_item['instructions'] is not None:  # Filter the wrong data
                                self.data.append(new_item)
                                scans.append(item['scan'])
        else:
            raise NotImplementedError

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self._load_adj_loc()

        self.env = EnvBatch(adj_loc_dict=self.adj_loc_dict, G_list=self.graphs)

        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_adj_loc(self):
        print('Loading adjacency locations')
        self.adj_loc_dict = json.load(open('data/adj_loc_dict.json', 'r'))

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)

        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, scanId, state_viewpointId, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state_viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[scanId][state_viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, scanId, viewpointId):
        long_id = "%s_%s" % (scanId, viewpointId)
        return self.adj_loc_dict[long_id]

    def _get_obs(self):
        obs = []
        for i, state in enumerate(self.env.getStates()):
            item = self.batch[i]
            # base_view_id = state['viewIndex']

            candidate = self.make_candidate(state['scanId'], state['viewpoint'])

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state['scanId'],
                'viewpoint' : state['viewpoint'],
                'viewIndex' : state['viewIndex'],
                'candidate': candidate,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state['scanId'], state['viewpoint'], item['path'][-1]),
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            })

        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        viewIds = [round(item['heading'] / math.pi * 6) % 12 + 12 for item in self.batch]
        # headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, viewIds)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats
