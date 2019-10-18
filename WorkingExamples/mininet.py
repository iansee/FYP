#!/usr/bin/python

import threading
from multiprocessing import Queue
import time
import random
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.cli import CLI


class SingleSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    def build( self, n=0 ):
        switch = self.addSwitch('s1')
        for h in range(n):
            host = self.addHost('h%s' % (h + 1))
            a = self.addLink( host, switch)
            print ('Added host %s' % (h + 1))


def perfTest():
    q = Queue()
    "Create network and run simple performance test"
    topo = SingleSwitchTopo( n=4 )
    net = Mininet( topo=topo, link=TCLink )
    net.start()
    commandprompt = threading.Thread(target=start_CLI, args=(net,))
    dynamiclinking = threading.Thread(target=linkfunc, args=(q,net,))
    commandprompt.start()
    dynamiclinking.start()
    commandprompt.join()
    q.put_nowait('1')
    dynamiclinking.join()
    net.stop()


def start_CLI(network):
    CLI(network)

def linkfunc(q,net):
    h1,h2,h3,h4 = net.get('h1','h2','h3','h4')
    while q.empty():
        time.sleep(30)
        #set randomized parameters here
        h4.intf().config(bw=1000,loss=20)
    return

    
if __name__ == '__main__':
    perfTest()
