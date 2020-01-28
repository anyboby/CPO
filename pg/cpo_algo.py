from functools import partial
from cpo_rl.pg.agent import CPOAgent
from cpo_rl.pg.run_agent import run_polopt_agent

def cpo(**kwargs):
    cpo_kwargs = dict(
                    reward_penalized=False,  # Irrelevant in CPO
                    objective_penalized=False,  # Irrelevant in CPO
                    learn_penalty=False,  # Irrelevant in CPO
                    penalty_param_loss=False  # Irrelevant in CPO
                    )
    agent = CPOAgent(**cpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)