from gym.envs.registration import register

register(
    id='LobotArmMoveSimple-v0', #5 action to the same goal
    entry_point='openai_ros2.envs:LobotArmMoveSimpleEnv',
)

register(
    id='LobotArmMoveRandom-v0', #5 action to random goal
    entry_point='openai_ros2.envs:LobotArmMoveSimpleRandomGoalEnv',
)

register(
    id='LobotArmMoveSimple-v1', #Con Action to the same goal
    entry_point='openai_ros2.envs:LobotArmMoveSimpleConActEnv',
)

register(
    id='LobotArmMoveRandom-v1', #Con Action to Random goal
    entry_point='openai_ros2.envs:LobotArmMoveRandomConActEnv',
)