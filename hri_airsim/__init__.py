from gym.envs.registration import register

register(
    id='HRI_AirSim-v0',
    entry_point='hri_airsim.envs:HRI_AirSim',
)

register(
    id='HRI_AirSim_Landing-v0',
    entry_point='hri_airsim.envs:HRI_AirSim_Landing',
)

