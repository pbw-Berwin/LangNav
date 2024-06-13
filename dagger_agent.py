import json
import os
from re import L
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from utils import padding_idx, print_progress
from collections import defaultdict
import math
import time

PROMPT_DICT = {
    "prompt_history": (
        "{head}\n\n"
        "### Current step:\n"
        "{current_step}\n\n"
        "### History:\n"
        "{history}\n\n"
        "### Response:"
    ),
    "prompt_no_history": (
        "{head}\n\n"
        "### Current step:\n"
        "{current_step}\n\n"
        "### This is the first step, no history available\n\n"
        "### Response:"
    ),
}

PROMPT_HISTORY_FIRST_DICT = {
    "prompt": (
        "{head}\n\n"
        "### Trajectory:\n"
        "{trajectory}\n"
        "You chose:\n"
    ),
}

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class DAggerAgent(BaseAgent):
    ''' An agent based on LLM. '''
    def __init__(self, llm, env, results_path, feat_dict, data_args, model_args):
        super(DAggerAgent, self).__init__(env, results_path)
        self.llm = llm
        self.feat_dict = feat_dict
        self.history_first = data_args.history_first
        self.fg_feature = data_args.fg_feature
        self.episode_len = model_args.maxAction
        self.policy_beta = model_args.policy_beta

        abbr2feature = {
            "c": "coordinates",
            "o": "obj_class",
            "d": "dense_cap"
        }
        if self.fg_feature != "":
            fg_feature_abbr = self.fg_feature.split("-")
            self.fg_feature = [abbr2feature[abbr] for abbr in fg_feature_abbr]
        else:
            self.fg_feature = []
            
        self.trajs = []

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)


    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = - torch.ones(len(obs)).long() * 100
        for i, ob in enumerate(obs):
            if not ended[i]:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = candidate['pointId']
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = -1
        return a

    def make_equiv_action(self, a_t, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """        
        new_obs = self.env.step(a_t)
        for i, ob in enumerate(new_obs):
            if a_t[i] != -1:            # -1 is the <stop> action
                if traj is not None:
                    traj[i]['path'].append((ob['viewpoint'],) + self.viewid2angle(ob['viewIndex']))

    def _init_prompt(self, obs):
        prompt_list = []
        prompt_template = (
            "You are a navigation agent who must navigate according to instructions given only descriptions of your current position via natural language. The natural language description is sometimes incorrect.\n\n"
            "### Instruction:\n"
            "{instruction}\n\n"
            "---\n\n"
        )
        for ob in obs:
            prompt = prompt_template.format(
                instruction=ob['instructions']
            )
            prompt_list.append(prompt)
        return prompt_list
    
    def reformat_prompt(self, prompt_list, ended):
        reformat_prompt_list = []
        for i, prompt in enumerate(prompt_list):
            head = prompt.split("\n\n---\n\n")[0]
            current = prompt.split("\n\n---\n\n")[-1]
            current = "\n".join(current.split("\n")[1:])

            if len(prompt.split("\n\n---\n\n")) > 2:
                history = prompt.split("\n\n---\n\n")[1:-1]
                history = "\n\n".join(history)
            else:
                history = None
            
            if history is not None:
                reformat_prompt_list.append(PROMPT_DICT["prompt_history"].format(head=head, current_step=current, history=history))
            else:
                reformat_prompt_list.append(PROMPT_DICT["prompt_no_history"].format(head=head, current_step=current))
        return reformat_prompt_list

    def reformat_history_first_prompt(self, prompt_list, ended):
        reformat_prompt_list = []
        for i, prompt in enumerate(prompt_list):
            head = prompt.split("\n\n---\n\n")[0]
            trajectory = prompt.split("\n\n---\n\n")[1:]
            trajectory = "\n\n".join(trajectory)
            reformat_prompt_list.append(PROMPT_HISTORY_FIRST_DICT["prompt"].format(head=head, trajectory=trajectory))
        return reformat_prompt_list

    def get_view_description(self, v2t_meta):
        if len(self.fg_feature) > 0 and len(v2t_meta['dense_cap']) > 0:
            dense_cap_template = []
            if "coordinates" in self.fg_feature:
                dense_cap_template.append("{coordinates}")
            if "obj_class" in self.fg_feature:
                dense_cap_template.append("{obj_class}")
            if "dense_cap" in self.fg_feature:
                dense_cap_template.append("{caption}")
            dense_cap_template = " - ".join(dense_cap_template)
            dense_cap_list = []
            for dense_cap in v2t_meta['dense_cap']:
                coords = dense_cap['coordinates'].split(' ')
                area = (float(coords[2]) - float(coords[0])) * (float(coords[3]) - float(coords[1]))
                confidence = float(dense_cap['confidence'])
                if area < 50 and confidence < 0.5:
                    continue
                dense_cap_list.append(dense_cap_template.format(**dense_cap))
            
            if len(dense_cap_list) > 0:
                description_template = (
                    "{global_cap}\n"
                    "Details:\n{dense_cap}"
                )
                dense_cap_list = list(set(dense_cap_list))
                return description_template.format(global_cap=v2t_meta['global_cap'], dense_cap="\n".join(dense_cap_list))
        return v2t_meta['global_cap']
    
    def append_obs(self, prompt_list, obs, ended, step):
        """
        Append the observation to the current observation
        :param obs: The observation
        :param ended: Whether the action seq is ended
        :return:
        """
        for i, ob in enumerate(obs):
            if not ended[i]:
                prompt = prompt_list[i]

                # assign the natural direction to each candidate
                direction_dict = {}
                for cand in ob['candidate']:
                    natural_direction = self.get_finegrained_direction_n_action(ob['viewIndex'], cand['pointId'])
                    if natural_direction not in direction_dict:
                        direction_dict[natural_direction] = [cand['pointId']]
                    else:
                        direction_dict[natural_direction].append(cand['pointId'])

                # generate the description
                panorama_description_list = []
                panorama_description_template = (
                    "{natural_direction}\n"
                    "{description}"
                )
                for natural_direction, pointIds in direction_dict.items():
                    # generate the description for each natural direction
                    view_descriptions = []
                    for pointId in pointIds:
                        v2t_meta = self.feat_dict[ob['scan']][ob['viewpoint']][str(pointId)]
                        view_dscpt = self.get_view_description(v2t_meta)
                        view_descriptions.append(view_dscpt)
                    # remove the duplicate descriptions
                    view_descriptions = list(set(view_descriptions))
                    view_descriptions = "\n".join(view_descriptions)
                    panorama_description_list.append(panorama_description_template.format(natural_direction=natural_direction, description=view_descriptions))
                
                panorama_description = "\n\n".join(panorama_description_list)        

                prompt += (
                    "Step {step}:\n\n"
                    "{panorama_description}\n"
                ).format(step=step, panorama_description=panorama_description)
                prompt_list[i] = prompt

        return prompt_list

    def append_action(self, prompt_list, action, obs, ended):
        """
        Append the action to the current observation
        :param action: The action
        :param obs: The observation
        :param ended: Whether the action seq is ended
        :return:
        """
        for i, ob in enumerate(obs):
            prompt = prompt_list[i]
            if action[i] == -1:
                prompt += (
                    "\nYou chose:\n"
                    "Stop\n\n"
                    "---\n\n"
                )
                prompt_list[i] = prompt
            elif action[i] == -100:
                assert ended[i]
            else:
                assert action[i] >= 0
                global_cap = self.feat_dict[ob['scan']][ob['viewpoint']][str(action[i].item())]['global_cap']
                prompt += (
                    "\nYou chose:\n"
                    "{global_cap}\n\n"
                    "---\n\n"
                ).format(global_cap=global_cap)
                prompt_list[i] = prompt
        return prompt_list

    def viewid2angle(self, viewid):
        viewid = int(viewid)
        heading = (viewid % 12) * math.pi / 6
        elevation = (viewid // 12) * math.pi / 6
        return heading, elevation
    
    def get_finegrained_direction_n_action(self, current_direction, candidate_direction):
        current_heading = current_direction % 12
        current_elevation = current_direction // 12

        target_heading = candidate_direction % 12
        target_elevation = candidate_direction // 12

        rel_heading = target_heading - current_heading
        rel_elevation = target_elevation - current_elevation
        
        if candidate_direction == -1:
            return "stop"

        if abs(rel_heading) > 6:
            if rel_heading > 0:
                rel_heading -= 12
            else:
                rel_heading += 12
        
        if rel_elevation > 0:
            elevation_toward = "{} degree up is, ".format(rel_elevation * 30)
        elif rel_elevation < 0:
            elevation_toward = "{} degree down is, ".format(abs(rel_elevation) * 30)
        else:
            elevation_toward = ""

        if abs(rel_heading) == 6:
            heading_toward = "To your back"
        elif rel_heading > 0:
            heading_toward = "To your {} degree right".format(rel_heading * 30)
        elif rel_heading < 0:
            heading_toward = "To your {} degree left".format(abs(rel_heading) * 30)
        else:
            heading_toward = "To your straight ahead"
        
        if elevation_toward != "":
            return heading_toward + " and " + elevation_toward
        return heading_toward + " is, "

    def get_action(self, ans_list, obs, ended, stop_if_not_found=True):
        a_t = - torch.ones(len(obs), dtype=torch.long) * 100
        for i, ob in enumerate(obs):
            if not ended[i]:
                found = False
                ans = ans_list[i]
                ans = ans.strip().lower() #[:-4]
                if '</s>' in ans:
                    ans = ans.replace('</s>', '')
                if ans == "stop":
                    a_t[i] = -1
                    found = True
                else:
                    for cand in ob['candidate']:
                        global_cap = self.feat_dict[ob['scan']][ob['viewpoint']][str(cand['pointId'])]['global_cap']
                        if ans == global_cap:                    
                            a_t[i] = int(cand['pointId'])
                            found = True
                            break
                if not found:
                    print('Answer not found: {}'.format(ans))
                    if stop_if_not_found:
                        a_t[i] = -1
                    else:
                        # random choose one from candidate's pointId and stop
                        action_space = [int(cand['pointId']) for cand in ob['candidate']] + [-1]
                        a_t[i] = random.choice(action_space)
                        print('Randomly choose: {}'.format(a_t[i]))
        return a_t

    def rollout(self, reset=True, collect_trajs=False, feedback='student', stop_if_not_found=True):
        """
        :param reset:       Reset the environment
        :return:
        """
        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        prompt_list = self._init_prompt(obs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'],) + self.viewid2angle(ob['viewIndex'])],
        } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        self.past_key_values = None

        start = time.time()
        for t in range(self.episode_len):

            prompt_list = self.append_obs(prompt_list, obs, ended=ended, step=t+1)

            if self.history_first:
                self.past_key_values = None
                reformat_prompt_list = self.reformat_history_first_prompt(prompt_list, ended=ended)
            else:
                self.past_key_values = None
                reformat_prompt_list = self.reformat_prompt(prompt_list, ended=ended)

            # import pdb; pdb.set_trace()
            ans_list, self.past_key_values = self.llm(reformat_prompt_list, ended, self.past_key_values, collect_mode=collect_trajs)

            a_t = self.get_action(ans_list, obs, ended, stop_if_not_found=stop_if_not_found)
            target = self._teacher_action(obs, ended)

            if collect_trajs:
                for prompt, gt, ob, if_end in zip(reformat_prompt_list, target, obs, ended):
                    if if_end:
                        continue
                    if gt.item() == -1:
                        ans = "stop"
                    else:
                        ans = self.feat_dict[ob['scan']][ob['viewpoint']][str(gt.item())]['global_cap']
                    self.trajs.append(
                        {'prompt': prompt, 'ans': ans}
                    )

            if feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif feedback == 'student':
                a_t = a_t        # student forcing - argmax
            elif feedback == 'sample':
                a_t = np.where(np.random.random(len(a_t)) < self.policy_beta, target, a_t)
            else:
                print(feedback)
                sys.exit('Invalid feedback option')
            
            prompt_list = self.append_action(prompt_list, a_t, obs, ended)

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            for i, next_id in enumerate(a_t):
                if ended[i]:    # The last action is <end>
                    a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(a_t, traj)
            obs = np.array(self.env._get_obs())

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        time_taken = time.time() - start
        len_of_traj = [len(traj[i]['path']) for i in range(len(traj))]

        print("Processed {} / {}".format(self.env.ix, len(self.env.data)))
        print("Average length of trajectory: {:.2f}".format(np.mean(len_of_traj)))
        print("Average time taken for each step: {:.2f} seconds".format(time_taken / np.mean(len_of_traj)))
        
        return traj

    def test(self, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        super(DAggerAgent, self).test(iters)

    def collect_trajs(self, output_file):
        self.trajs = []
        self.results = {}
        self.env.reset_epoch(shuffle=True)
        # We rely on env showing the entire batch before repeating anything
        looped = False
        while True:
            sampled_trajs = self.rollout(reset=True, collect_trajs=True, feedback='sample', stop_if_not_found=False)
            for traj in sampled_trajs:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break
        
        print('Collected {} trajectories'.format(len(self.trajs)))
        with open(output_file, 'w') as f:
            json.dump(self.trajs, f)
        print('Saved to {}'.format(output_file))