import gym
import time
import time

if __name__ == '__main__':
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
    cart = viewer.draw_line((10,10), (200, 200))
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
    viewer.render(return_rgb_array = False)

    time.sleep(5)
    viewer.close()
