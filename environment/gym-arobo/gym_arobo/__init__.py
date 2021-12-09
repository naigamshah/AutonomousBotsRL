from gym.envs.registration import register

register(id='ARobo-v0', 
    entry_point='gym_arobo.envs:ARoboEnv', 
)
