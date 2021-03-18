
from environment import Easy21Env,HIT,STICK
from mc import MCControl
from td import TDControl
from tdl import TDLControl
from function import FunctionControl

if __name__ == '__main__':
    env = Easy21Env()
    # mc = MCControl(env, env.get_num_states(), [HIT,STICK])
    # mc.control()

    # tdl = TDLControl(env, env.get_num_states(), [HIT,STICK])
    # tdl.control()

    f = FunctionControl(env, [HIT,STICK])
    f.control()


