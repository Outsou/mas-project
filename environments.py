from creamas.mp import MultiEnvironment
from creamas.util import run

import os
import shutil
import aiomas


class StatEnvironment(MultiEnvironment):
    '''A MultiEnvironment that can collect stats from agents.'''

    def cause_change(self, amount):
        agents = self.get_agents(addr=False)
        for agent in agents:
            run(agent.cause_change(amount))

    def get_connection_counts(self):
        return self.get_dictionary('get_connection_counts')

    def get_comparison_counts(self):
        return self.get_dictionary('get_comparison_count')

    def get_artifacts_created(self):
        return self.get_dictionary('get_artifacts_created')

    def get_passed_self_criticism_counts(self):
        return self.get_dictionary('get_passed_self_criticism_count')

    def get_recommendations(self):
        return self.get_dictionary('get_recommendations')

    def get_total_rewards(self):
        return self.get_dictionary('get_total_reward')

    def get_dictionary(self, func_name):
        '''
        Creates a dictionary with agents as the keys
        and the return values of a function as the values.

        :param func_name:
            The name of the function that will be called for each agent.
        :return:
            A dictionary.
        '''
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = aiomas.run(until=func())

        return dict

    def save_artifacts(self, folder):
        '''
        Asks each agent to save their artifacts.

        :param folder:
            The folder where the artifacts are saved.
        '''
        def agent_name_parse(name):
            '''Converts the name of an agent into a file path friendly format.'''
            parsed_name = name.replace('://', '_')
            parsed_name = parsed_name.replace(':', '_')
            parsed_name = parsed_name.replace('/', '_')
            return parsed_name

        agents = self.get_agents(addr=False)
        for agent in agents:
            name = run(agent.get_name())
            agent_folder = '{}/{}'.format(folder, agent_name_parse(name))
            if os.path.exists(agent_folder):
                shutil.rmtree(agent_folder)
            os.makedirs(agent_folder)
            artifacts, artifact_cls = run(agent.get_artifacts())
            for i in range(len(artifacts)):
                artifact_cls.save_artifact(artifacts[i], agent_folder, i, artifacts[i].evals[name])
