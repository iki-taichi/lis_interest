# -*- coding: utf-8 -*-

import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from cnn_dqn_agent import CnnDqnAgent
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np
import multiprocessing as mp

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help='reward log file name')
args = parser.parse_args()

def agent_process(gpu_id, log_file, q_from_parent, q_to_parent):
    # initialization
    depth_image_dim = 32*32
    depth_image_count = 1
    
    has_started = False
    cycle_counter = 0
    reward_sum = 0
    agent = CnnDqnAgent()
        
    print("initializing agent...")
    agent.agent_init(
            use_gpu=gpu_id,
            depth_image_dim=depth_image_dim*depth_image_count,
        )
            
    with open(log_file, 'w') as the_file:
        the_file.write('cycle, episode_reward_sum \n')
    
    # step
    byte_data = q_from_parent.get()
    while not byte_data is None:
            #try:
            # data extraction
            dat = msgpack.unpackb(byte_data)
            image = [
                    Image.open(io.BytesIO(bytearray(dat['image'][i])))
                    for i in xrange(depth_image_count)
                ]
            depth = [
                    np.array(ImageOps.grayscale(Image.open(io.BytesIO(bytearray(dat['depth'][i]))))).reshape(depth_image_dim) 
                    for i in xrange(depth_image_count)
                ]
            observation = {"image": image, "depth": depth}
            reward = dat['reward']
            end_episode = dat['endEpisode']
        
            # action-making
            ret = None
            if not has_started:
                has_started = True
                ret = agent.agent_start(observation)
            else:
                cycle_counter += 1
                reward_sum += reward

                if end_episode:
                    agent.agent_end(reward)
                    with open(log_file, 'a') as the_file:
                        the_file.write('%d, %f\n'%(cycle_counter, reward_sum))
                    reward_sum = 0
                
                    ret = agent.agent_start(observation) 
                else:
                    action, eps, q_now, new_feature_vec, deg_interest = agent.agent_step(reward, observation)
                    agent.agent_step_update(reward, action, eps, q_now, new_feature_vec, deg_interest)
                    ret = (action, deg_interest)
        
            q_to_parent.put(ret)
            #except Exception as e:
            #print(e)
            #q_to_parent.put(None)
            #raise e
            byte_data = q_from_parent.get()
    
class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler        

class AgentServer(WebSocket):
    process_info = {}
    
    def send_action(self, ret):
        dat = msgpack.packb({
                "command": str(ret[0]),
                "degInterest": str(ret[1]),
            })
        self.send(dat, binary=True)

    def received_message(self, m):
        # when you use multi agents, you can specify agent by name
        agent_name = 'agent1'
        pinfo = self.process_info.get(agent_name, None)
        if pinfo is None:
            q_from_parent = mp.Queue()
            q_to_parent = mp.Queue()
            p = mp.Process(target=agent_process, args=(args.gpu, args.log_file, q_from_parent, q_to_parent))
            p.daemon = True
            p.start()
            pinfo = (
                    agent_name,    # 0
                    q_from_parent, # 1
                    q_to_parent,   # 2
                    p,             # 3
                )
            self.process_info[agent_name] = pinfo
        
        pinfo[1].put(m.data)
        action = pinfo[2].get()
        self.send_action(action)

if __name__ == '__main__':
    cherrypy.config.update({
            'server.socket_host': args.ip,
            'server.socket_port': args.port
        })
    WebSocketPlugin(cherrypy.engine).subscribe()
    cherrypy.tools.websocket = WebSocketTool()
    cherrypy.config.update({'engine.autoreload.on': False})
    config = {
            '/ws': {'tools.websocket.on': True, 'tools.websocket.handler_cls': AgentServer}
        }
    cherrypy.quickstart(Root(), '/', config)
