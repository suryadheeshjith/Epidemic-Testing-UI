import random

class agent_policy_based_lockdown():
    def __init__(self,policy_to_consider,value_list,do_lockdown_fn,time_period):
        self.policy_type='Restrict'
        self.policy_to_consider = policy_to_consider
        self.do_lockdown_fn=do_lockdown_fn
        self.value_list=value_list
        self.time_period = time_period

    def enact_policy(self,time_step,agents):
        for agent in agents:
            history = agent.testing_hist
            if(len(history)):
                last_time_step = history[-1].time_step
                if(time_step - last_time_step < self.time_period):
                    result = self.get_accumulated_result(history,last_time_step)
                    if(result in self.value_list):

                        if(time_step - history[-1].machine_start_step==history[-1].turnaround_time):
                            if(agent.state!="Infected"):
                                agent.quarantine_list.append("Wrong")

                            else:
                                agent.quarantine_list.append("Right")

                        else:

                            agent.quarantine_list.append(agent.quarantine_list[-1])

                        agent.quarantined= True

    def get_accumulated_result(self,history,last_time_step):

        indx = len(history)-1
        while(indx>=0 and history[indx].time_step==last_time_step):
            if(history[indx].result == "Negative"):
                return "Negative"

            indx-=1

        return "Positive"
