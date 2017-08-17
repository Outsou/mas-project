from creamas.mp import MultiEnvironment
from creamas.util import run

import os
import shutil
import aiomas


class StatEnvironment(MultiEnvironment):
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
        agents = self.get_agents(addr=False)

        dict = {}

        for agent in agents:
            name = aiomas.run(until=agent.get_name())
            func = getattr(agent, func_name)
            dict[name] = aiomas.run(until=func())

        return dict

    def save_artifacts(self, folder):
        def agent_name_parse(name):
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
            run(agent.save_artifacts(agent_folder))
