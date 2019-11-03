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
    def build( self, nodes = 0, bad_nodes = 0, upperbound = 0, lowerbound = 0):
        switch = self.addSwitch('s1')
        host = self.addHost('h1')
        a = self.addLink( host, switch)
        print ('Added host h1')

        counter = 0
        bad_node_list = []
        while (counter != bad_nodes):
            bad_host = random.randint(1,nodes)
            if bad_host not in bad_node_list:
                bad_node_list.append(bad_host)
                counter+=1

        for h in range(1, nodes):
            switch1 = self.addSwitch('s%s' % (h + 1))
            host = self.addHost('h%s' % (h + 1))
            if h in bad_node_list:
                random_loss = random.randint(lowerbound,upperbound)
                print ("h{} has been choosen as a faulty host with a loss of {}".format(
                    h+1,random_loss))
                a = self.addLink( switch, switch1, loss=random_loss)
            else:
                print ("h{} has been choosen as a good host".format(
                    h+1))
                a = self.addLink( switch, switch1)
            a = self.addLink( host, switch1)
            print ('Added host h%s' % (h + 1))


def runcommand(number,network):
    name = 'h%s'%number
    host = network.get(name)
    print ('Created Federated Client on Host {}'.format(name))
    command = ('python /home/mininet/Federated_Swarm.py %s' %(number))
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

    topo = SingleSwitchTopo(nodes, bad_nodes, upperbound, lowerbound)
    net = Mininet( topo=topo, link=TCLink )
    net.start()
    commandprompt = threading.Thread(target=start_CLI, args=(net,))
    commandprompt.start()
    time.sleep(5)
    print ('\n')

    # print ('Starting varying packet loss')
    # linkfunc(net,upperbound,lowerbound,bad_nodes,nodes)

    for i in range(2,nodes+1):
        runcommand(i,net)


    commandprompt.join()

if __name__ == '__main__':
    main()
