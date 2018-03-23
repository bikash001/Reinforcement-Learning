import gym
import time
import time
import numpy as np


def margin(els, val):
    return [(els[0][0]+val, els[0][1]+val),
            (els[1][0]+val, els[1][1]-val),
            (els[2][0]-val, els[2][1]-val),
            (els[3][0]-val, els[3][1]+val)]

if __name__ == '__main__':
    

    state_rewards = np.zeros((12,12))
    for i in range(3,10):
        state_rewards[i][3] = -1
    for i in range(3, 9):
        state_rewards[9][i] = -1
    for i in range(5, 10):
        state_rewards[i][8] = -1
    for i in range(3, 6):
        state_rewards[i][7] = -1
    for i in range(3, 8):
        state_rewards[3][i] = -1

    for i in range(4,9):
        state_rewards[i][4] = -1
    for i in range(4, 8):
        state_rewards[8][i] = -1
    for i in range(6, 9):
        state_rewards[i][7] = -1
    for i in range(4, 7):
        state_rewards[i][6] = -1
    for i in range(4, 7):
        state_rewards[4][i] = -1

    for i in range(5,8):
        state_rewards[i][5] = -1
    state_rewards[7][6]


    screen_width = 500
    screen_height = 500

    world_width = 12
    scale = screen_width/world_width
    # carty = 100 # TOP OF CART
    # polewidth = 10.0
    # polelen = scale * 1.0
    # cartwidth = 50.0
    # cartheight = 30.0

    from gym.envs.classic_control import rendering
    viewer = rendering.Viewer(screen_width, screen_height)
    l,r,t,b = 0,500,500,0
    # l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
    # axleoffset =cartheight/4.0
    # cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    # cart = viewer.draw_line((10,10), (200, 200))
    # carttrans = rendering.Transform()
    # cart.add_attr(carttrans)
    # cart.set_color(1.,1.,1.)
    # viewer.add_geom(cart)
    # l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
    # pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    # pole.set_color(.8,.6,.4)
    # poletrans = rendering.Transform(translation=(0, axleoffset))
    # pole.add_attr(poletrans)
    # pole.add_attr(carttrans)
    # viewer.add_geom(pole)
    # axle = rendering.make_circle(polewidth/2)
    # axle.add_attr(poletrans)
    # axle.add_attr(carttrans)
    # axle.set_color(.5,.5,.8)
    # viewer.add_geom(axle)
    # track = rendering.Line((0,carty), (screen_width,carty))
    # track.set_color(0,0,0)
    # viewer.add_geom(track)

    # if state is None: return None

    # x = state
    # cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
    # carttrans.set_translation(cartx, carty)
    # poletrans.set_rotation(-x[2])
    
    l = np.linspace(10, screen_height-10, 13)
    w = np.linspace(10, screen_width-10, 13)

    arr = []
    for i in range(len(l)-1):
        a = []
        for j in range(len(w)-1):
            a.append([(l[i], w[j]), (l[i], w[j+1]), (l[i+1], w[j+1]), (l[i+1], w[j])])

        arr.append(a)

    # make grid
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            # check if obstacle
            if state_rewards[i][j] == 0:
                cart = rendering.make_polygon(arr[i][j], False)
                cart.set_color(0,0,0)
            elif state_rewards[i][j] == -1:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0.9,0.9,0.9)
            elif state_rewards[i][j] == -2:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0.6,0.6,0.6)
            elif state_rewards[i][j] == -3:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0,0,0)
            viewer.add_geom(cart)


    # agent position
    cart = rendering.FilledPolygon(margin(arr[2][3], 10))
    cart.set_color(1.,0,0)
    viewer.add_geom(cart)    


    viewer.render(return_rgb_array = False)

    time.sleep(5)
    viewer.close()
