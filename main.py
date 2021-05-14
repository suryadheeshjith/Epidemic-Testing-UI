import random
import copy
import streamlit as st
import numpy as np
import pandas as pd
import Testing_Policy
import Lockdown_Policy

class Agent():
    def __init__(self,state,index):
        self.state=state
        self.index=index
        self.neighbours=[]
        self.quarantined=False

        self.testing_hist=[]
        self.quarantine_list = []


class RandomGraph():
    def __init__(self,n,p,connected=True):
        self.n=n
        self.p=p
        self.connected=connected

        self.adj_list=[]
        for i in range(n):
            self.adj_list.append([])

        for i in range(n):
            for j in range(i+1,n):
                if random.random()<p:
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)

        if self.connected:
            for i in range(n):
                if self.adj_list[i]==[]:
                    allowed_values = list(range(0, n))
                    allowed_values.remove(i)
                    j = random.choice(allowed_values)
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)

        dsum=0
        for i in range(n):
            dsum+=len(self.adj_list[i])
        self.average_degree=2*dsum/n

class Simulate():
    def __init__(self,graph_obj,agents,transmission_probability,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown):
        self.graph_obj=graph_obj
        self.agents=agents
        self.transmission_probability=transmission_probability
        self.state_list={}
        self.state_history={}
        self.quarantine_list=[]
        self.quarantine_history=[]

        # Testing
        self.machines = machines
        self.option = option
        self.testing_methods_list = testing_methods_list
        self.num_agents_per_step = num_agents_per_step
        self.num_days_lockdown = num_days_lockdown


        for state in transmission_probability.keys():
            self.state_list[state]=[]
            self.state_history[state]=[]

        for agent in self.agents:
            self.state_list[agent.state].append(agent.index)

        self.update()
        self.init_testing()

    def update(self):
        for state in self.state_history.keys():
            self.state_history[state].append(len(self.state_list[state]))
        self.quarantine_history.append(len(self.quarantine_list))

    def init_testing(self):

        if(self.machines):
            self.testing_policy = Testing_Policy.Test_Policy(lambda x:self.num_agents_per_step)
            for machine in self.machines.keys():
                self.testing_policy.add_machine(machine,machines[machine]["cost"],machines[machine]["false_positive_rate"],\
                                                machines[machine]["false_negative_rate"],machines[machine]["turnaround_time"],machines[machine]["capacity"])



            if(self.option=="Normal Testing"):
                self.testing_policy.set_register_agent_testtube_func(self.testing_policy.random_agents(1,1))

            elif(self.option=="Group Testing"):
                num_agents_per_test = self.testing_methods_list[option]["num_agents_per_test"]
                num_tests_per_agent = self.testing_methods_list[option]["num_tests_per_agent"]
                self.testing_policy.set_register_agent_testtube_func(self.testing_policy.random_agents(num_agents_per_test,num_tests_per_agent))

            elif(self.option=="Friendship Testing"):
                min_days = self.testing_methods_list[option]["min_days"]
                self.testing_policy.set_register_agent_testtube_func(self.testing_policy.friendship_testing(min_days))


            self.lockdown_policy = Lockdown_Policy.agent_policy_based_lockdown("Testing",["Positive"],lambda x:True,self.num_days_lockdown)


    def simulate_days(self,days):
        for day in range(days):
            self.simulate_day(day)

    def simulate_day(self,day):
        self.new_day()
        self.enact_policy(day)
        self.spread(day)
        self.update()

    def new_day(self):
        for agent in self.agents:
            agent.quarantined = False

    def enact_policy(self,day):
        if(self.machines):
            self.testing_policy.enact_policy(day,self.agents)
            self.lockdown_policy.enact_policy(day,self.agents)


    def spread(self,day):
        #Inf : Sus ->2bExp
        to_beExposed=[]
        for agent_indx in self.state_list['Susceptible']:
            agent=self.agents[agent_indx]
            p_inf=self.transmission_probability['Susceptible']['Exposed'](agent,agent.neighbours)
            if random.random()<p_inf:
                to_beExposed.append(agent_indx)
        # Inf ->Rec
        for agent_indx in self.state_list['Infected']:
            agent=self.agents[agent_indx]
            p=self.transmission_probability['Infected']['Recovered'](agent,agent.neighbours)
            self.convert_state(agent,'Recovered',p)

        # Exp -> Inf
        for agent_indx in self.state_list['Exposed']:
            agent=self.agents[agent_indx]
            p1=self.transmission_probability['Exposed']['Infected'](agent,agent.neighbours)
            self.convert_state(agent,'Infected',p1)

        #2bExp->Exp
        for agent_indx in to_beExposed:
            agent=self.agents[agent_indx]
            self.convert_state(agent,'Exposed',1)

    def convert_state(self,agent,new_state,p):
        if random.random()<p:
            self.state_list[agent.state].remove(agent.index)
            agent.state=new_state
            self.state_list[agent.state].append(agent.index)

    def get_total_machine_cost(self):
        total = 0
        for machine in self.testing_policy.machine_list:
            total+=machine.total_cost
        return total



def world(n,p,inf_per,days,graph_obj,beta,mu,gamma,delta,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown):
    #print("Average degree : ",graph_obj.average_degree)
    agents=[]
    for i in range(n):
        state='Susceptible'
        if random.random()<inf_per:
            state='Infected'
        agent=Agent(state,i)
        agents.append(agent)

    #create graph of agents from graph_obj
    for indx,agent in enumerate(agents):
        agent.index=indx
        for j in graph_obj.adj_list[indx]:
            agent.neighbours.append(agents[j])

    individual_types=['Susceptible','Exposed','Infected','Recovered']

    def p_infection(p_inf):  # probability of infectiong neighbour
        def p_fn(my_agent,neighbour_agents):
            p_not_inf=1
            for nbr_agent in neighbour_agents:
                if nbr_agent.state=='Infected' and not nbr_agent.quarantined and not my_agent.quarantined:
                    p_not_inf*=(1-p_inf)
            return 1 - p_not_inf
        return p_fn


    def p_standard(p):
        def p_fn(my_agent,neighbour_agents):
            return p
        return p_fn

    transmission_prob={}
    for t in individual_types:
        transmission_prob[t]={}

    for t1 in individual_types:
        for t2 in individual_types:
            transmission_prob[t1][t2]=p_standard(0)
    transmission_prob['Susceptible']['Exposed']= p_infection(beta)
    transmission_prob['Exposed']['Infected']= p_standard(mu)
    transmission_prob['Infected']['Recovered']= p_standard(gamma)
    transmission_prob['Recovered']['Susceptible']= p_standard(delta)

    sim_obj=Simulate(graph_obj,agents,transmission_prob,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown)
    sim_obj.simulate_days(days)
    return sim_obj.state_history, agents, sim_obj.get_total_machine_cost()

def average(tdict,number):
    for k in tdict.keys():
        l=tdict[k]
        for i in range(len(l)):
            tdict[k][i]/=number

    return tdict

def worlds(number,n,p,inf_per,days,beta,mu,gamma,delta,latest_iteration,bar,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown):

    individual_types=['Susceptible','Exposed','Infected','Recovered']
    tdict={}
    total_quarantined_days = 0
    wrongly_quarantined_days = 0
    total_infection = 0
    total_machine_cost = 0

    for state in individual_types:
        tdict[state]=[0]*(days+1)

    for i in range(number):
        latest_iteration.text('Simulating World : {0}'.format(i+1))
        bar.progress(i + 1)
        graph_obj = RandomGraph(n,p,True)
        sdict,agents,machine_cost = world(n,p,inf_per,days,graph_obj,beta,mu,gamma,delta,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown)
        total_machine_cost += machine_cost

        for agent in agents:
            for truth in agent.quarantine_list:
                if(truth=="Right"):
                    total_quarantined_days+=1
                elif(truth=="Wrong"):
                    total_quarantined_days+=1
                    wrongly_quarantined_days+=1

        total_infection+=len(agents)-sdict["Susceptible"][-1]

        for state in individual_types:
            for j in range(len(tdict[state])):
                tdict[state][j]+=sdict[state][j]

    tdict=average(tdict,number)

    total_infection /=number
    total_quarantined_days /=number
    wrongly_quarantined_days/=number
    total_machine_cost/=number

    print("Total Infections : ",total_infection)
    print("Total quarantined days : ",total_quarantined_days)
    print("Wrongly quarantined days : ",wrongly_quarantined_days)
    print("Total Test cost : ",total_machine_cost)

    values=[]
    keys=individual_types

    for i in range(len(tdict['Susceptible'])):
        values.append([])
        for state in individual_types:
            values[-1].append(tdict[state][i])

    chart_data = pd.DataFrame(
    values,
    columns=keys)
    st.line_chart(chart_data)

    return total_infection, total_quarantined_days, wrongly_quarantined_days, total_machine_cost

if __name__=="__main__":
    random.seed(0)
    st.write("""
    # Testing Policy
    This simulator allows you to test different testing policies with an SEIRS model on a G(n,p) random graph. We have designed this application
    as a simple point solution of the main simulator - [Episimmer](https://github.com/healthbadge/episimmer) to showcase it's functionality. Please
    note that the main simulator can do a lot more than what is on this website right now and we made this application only for users to get a general
    idea.

    Anyway, Use the sidebar to set parameters. Have Fun :)
    """)
    st.write("------------------------------------------------------------------------------------")

    latest_iteration = st.text("Ready!")
    bar = st.progress(0)

    st.sidebar.write("World parameters")
    n=st.sidebar.slider("Number of agents", min_value=0 , max_value=5000 , value=1000 , step=100 , format=None , key=None )
    #st.sidebar.text("Number of agents selected : {0}".format(n))
    # p=st.sidebar.select_slider("Probability(p) of an edge in G(n,p) random graph", value = 0.1, options = list(np.linspace(0,1,1001)))
    p=st.sidebar.slider("Probability(p) of an edge in G(n,p) random graph",  min_value=0.0 , max_value=1.0 , value=0.1 , step=0.01)
    p_range=st.sidebar.checkbox("Divide p by 10")
    if p_range:
        p/=10
    #st.sidebar.text("Probability selected : {0}".format(p))

    days=st.sidebar.slider("Number of days in simulation", min_value=1 , max_value=100 , value=30 , step=1 , format=None , key=None )
    #st.sidebar.text("Number of days selected : {0}".format(days))
    num_worlds=st.sidebar.slider("Number of times to average simulations over", min_value=1 , max_value=100 , value=1 , step=1 , format=None , key=None )
    #st.sidebar.text("Number of simulations selected: {0}".format(num_worlds))

    st.sidebar.write("------------------------------------------------------------------------------------")

    machines = {}
    testing_methods_list = {"Normal Testing":{}, "Group Testing":{}, "Friendship Testing":{}}
    option = None
    num_days_lockdown = 0

    st.sidebar.write("Testing parameters")
    num_agents_per_step=st.sidebar.slider("Number of Agents to test every day", min_value=0 , max_value=1000 , value=100 , step=10 , format=None , key=None )
    #st.sidebar.text("Number of agents selected : {0}".format(num_agents_per_step))
    num_distinct_tests=st.sidebar.slider("Number of distinct tests", min_value=0 , max_value=10 , value=1 , step=1 , format=None , key=None)
    #st.sidebar.text("Number of distict tests selected : {0}".format(num_distinct_tests))
    for i in range(num_distinct_tests):
        st.sidebar.text("Test Type {0}".format(i+1))
        machines['Test'+str(i+1)] = {}

        cost = st.sidebar.slider("Cost of test", min_value=1 , max_value=1000 , value=1 , step=1, key=i)
        #st.sidebar.text("Cost selected : {0}".format(cost))
        false_positive_rate=st.sidebar.slider("False Positive Rate", min_value=0.0 , max_value=1.0 , value=0.0 , step=0.01, key=i)
        #st.sidebar.text("False Positive Rate selected : {0}".format(false_positive_rate))
        false_negative_rate=st.sidebar.slider("False Negative Rate", min_value=0.0 , max_value=1.0 , value=0.0 , step=0.01, key=i)
        #st.sidebar.text("False Negative Rate selected : {0}".format(false_negative_rate))
        turnaround_time=st.sidebar.slider("Turnaround time (Steps for the test to complete)", min_value=0 , max_value=100 , value=0 , step=1, key=i)
        #st.sidebar.text("Turnaround time selected : {0}".format(turnaround_time))
        capacity=st.sidebar.slider("Maximum tests done by Test {0} per day".format(i+1), min_value=1 , max_value=1000 , value=20 , step=1, key=i)
        #st.sidebar.text("Maximum tests selected : {0}".format(capacity))

        machines['Test'+str(i+1)]['cost'] = cost
        machines['Test'+str(i+1)]['false_positive_rate'] = false_positive_rate
        machines['Test'+str(i+1)]['false_negative_rate'] = false_negative_rate
        machines['Test'+str(i+1)]['turnaround_time'] = turnaround_time
        machines['Test'+str(i+1)]['capacity'] = capacity

    if(num_distinct_tests):
        option = st.sidebar.radio('Choose Testing Method',list(testing_methods_list.keys()))

        if(option=="Normal Testing"):
            testing_methods_list[option] = True

        elif(option=="Group Testing"):
            num_agents = st.sidebar.slider("Number of agents per test", min_value=1 , max_value=15 , value=1 , step=1)
            #st.sidebar.text("Agents per test selected : {0}".format(num_agents))
            num_tests = st.sidebar.slider("Number of tests per agent", min_value=1 , max_value=15 , value=1 , step=1)
            #st.sidebar.text("Tests per agent selected : {0}".format(num_tests))

            testing_methods_list[option]["num_agents_per_test"] = num_agents
            testing_methods_list[option]["num_tests_per_agent"] = num_tests

        elif(option=="Friendship Testing"):
            min_days = st.sidebar.slider("Minimum days for agent to test again", min_value=1 , max_value=100 , value=5 , step=1)
            #st.sidebar.text("Minimum days selected : {0}".format(min_days))
            testing_methods_list[option]["min_days"] = min_days

        st.sidebar.write("Lockdown parameters")
        num_days_lockdown = st.sidebar.slider("Number of days to lockdown agent when tested positive", min_value=0 , max_value=100 , value=5 , step=1)
        #st.sidebar.text("Lockdown days selected : {0}".format(num_days_lockdown))
    else:
        no_tests_text = st.sidebar.empty()
        no_tests_text.text("No testing done!")


    st.sidebar.write("------------------------------------------------------------------------------------")

    st.sidebar.write("Disease parameters")
    # inf_per=0.01
    inf_per = st.sidebar.slider("Initial prevalence of infection", min_value=0.0 , max_value=1.0 , value=0.01 , step=0.01 , format=None , key=None )

    beta=st.sidebar.slider("Rate of infection : Susceptible->Exposed", min_value=0.0 , max_value=1.0 , value=0.2 , step=0.01 , format=None , key=None )
    #st.sidebar.text("Rate of infection selected: {0}".format(beta))
    mu=st.sidebar.slider("Rate of Exposed->Infected", min_value=0.0 , max_value=1.0 , value=0.2 , step=0.01 , format=None , key=None )
    #st.sidebar.text("Rate selected: {0}".format(mu))
    gamma=st.sidebar.slider("Rate of recovery : Infected:->Recovered", min_value=0.0 , max_value=1.0 , value=0.2 , step=0.01 , format=None , key=None )
    #st.sidebar.text("Rate of recovery selected: {0}".format(gamma))
    delta=st.sidebar.slider("Rate of unimmunisation : Recovered->Susceptible", min_value=0.0 , max_value=1.0 , value=0.0 , step=0.01 , format=None , key=None )
    #st.sidebar.text("Rate of unimmunisation selected: {0}".format(delta))

    st.sidebar.write("------------------------------------------------------------------------------------")

    total_infection, total_quarantined_days,\
    wrongly_quarantined_days, total_machine_cost = worlds(num_worlds,n,p,inf_per,days,beta,mu,gamma,delta,latest_iteration,bar,machines,testing_methods_list,option,num_agents_per_step,num_days_lockdown)

    latest_iteration.text('Ready!')
    bar.progress(0)
    st.write("------------------------------------------------------------------------------------")

    st.header("Cost")
    st.write("Goal is to minimise the cost function :")
    st.latex(r'''Cost\ function =  a \times Cumulative\ Infected\ persons + b \times Cumulative\ Quarantined\ Days \\+ Cumulative\ Cost\ of\ Tests\ per\ day''')
    st.write(" -- 'a' refers to medical cost per infected per day")
    st.write(" -- 'b' refers to economic loss of lockdown of an agent per day")

    st.write("------------------------------------------------------------------------------------")

    a=st.slider("Medical cost per infected per day", min_value=0 , max_value=100 , value=1 , step=1 , format=None , key=None )
    b=st.slider("Economic loss during lockdown per individual per day", min_value=0 , max_value=100 , value=1 , step=1 , format=None , key=None )

    st.write("The Cumulative cost is "+str(a*total_infection+total_machine_cost+b*(total_quarantined_days)))
