#!/usr/bin/python

import time
import concurrent.futures
from random import random
import random
import threading
from multiprocessing import Process
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

    
def runcommand(number,network):
    name = 'h%s'%number
    host = network.get(name)
    print ('Created Federated Client on Host {}'.format(name))
    command = ('python /home/mininet/imported_files/Federated_Swarm.py %s' %(number))
    host.sendCmd(command)

def start_CLI(network):
    CLI(network)
    

def linkfunc(net,upperbound,lowerbound,bad_nodes,nodes):
    counter = 0
    bad_node_list = []
    while (counter != bad_nodes):
        random_no = random.randint(2,nodes)
        bad_host = 'h{}'.format(random_no)
        if bad_host not in bad_node_list:
            bad_node_list.append(bad_host)
            counter+=1
    
    for bad_host in bad_node_list:
        this_bad_host = net.get(bad_host)
        random_loss = random.randint(lowerbound,upperbound)
        print ("{} has been choosen as a faulty host with a loss of {}".format(
            bad_host,random_loss))

        this_bad_host.intf().config(loss=random_loss)

            
def main():
    print ("Use H1 as the centralized node")
    nodes = int(input("Enter amount of nodes:"))
    bad_nodes = int(input("Enter amount of nodes with faulty connection:"))
    upperbound = int(input("Enter the upper bound for loss:"))
    lowerbound = int(input("Enter the lower bound for loss:"))
    
    topo = SingleSwitchTopo(nodes)
    net = Mininet( topo=topo, link=TCLink )
    net.start()
    commandprompt = threading.Thread(target=start_CLI, args=(net,))
    commandprompt.start()
    time.sleep(5)
    print ('\n')
    
    print ('Starting varying packet loss')
    linkfunc(net,upperbound,lowerbound,bad_nodes,nodes)
    
    for i in range(2,nodes+1):
        runcommand(i,net)
        

    commandprompt.join()
            
if __name__ == '__main__':
    main()
