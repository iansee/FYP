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


#Hard coded node spawning
#2,3,4 -> 2,3,5 0%
#5,6,7 -> 2,3,5 5%
#8,9,10 -> 2,3,5 10%

class SingleSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    def build( self, n=0 ):
        switch = self.addSwitch('s1')
        for h in range(n):
            host = self.addHost('h%s' % (h + 1))
            a = self.addLink( host, switch)
            print ('Added host %s' % (h + 1))

    
def runcommand(number, startslice,endslice,network):
    name = 'h%s'%number
    host = network.get(name)
    print ('Created Federated Client on Host {}'.format(name))
    command = ('FILEPATH/Federated_Swarm.py %s %s %s' %(number,startslice,endslice))
    host.sendCmd(command)

def start_CLI(network):
    CLI(network)

def linkfunc(net):
    switch = net.get('s1')
    nodelist = []
    for i in range (2,11):
        this_node = 'h{}'.format(i)
        this_node_n = net.get(this_node)
        linkslist = net.linksBetween(switch,this_node_n)
        thislink = linkslist[0]
        if i <= 4: #NODES 2,3,4
            thislink.intf1.config(loss=0)
            thislink.intf2.config(loss=0)
            print ("{} has been choosen as a faulty host with a loss of {}".format(this_node,0))

        elif 5 <= i <= 7: #NODES 5,6,7
            thislink.intf1.config(loss=5)
            thislink.intf2.config(loss=5)
            print ("{} has been choosen as a faulty host with a loss of {}".format(this_node,5))
        else:#NODES 8,9,10
            thislink.intf1.config(loss=10)
            thislink.intf2.config(loss=10)
            print ("{} has been choosen as a faulty host with a loss of {}".format(this_node,10))

        
            
    '''
    counter = 0
    bad_node_list = []
    while (counter != bad_nodes):
        random_no = random.randint(2,nodes)
        bad_host = 'h{}'.format(random_no)
        if bad_host not in bad_node_list:
            bad_node_list.append(bad_host)
            counter+=1
            
    switch = net.get('s1')
    for bad_host in bad_node_list:
        this_bad_host = net.get(bad_host)
        linkslist = net.linksBetween(switch,this_bad_host)
        thislink = linkslist[0] #Get link which has the 2 interfaces
        

        random_loss = random.randint(lowerbound,upperbound)
        print ("{} has been choosen as a faulty host with a loss of {}".format(
            bad_host,random_loss))
        
        this_inf1 = thislink.intf1 #set interface 1
        this_inf1.config(loss=random_loss)
        this_inf2 = thislink.intf2 #set interface 2
        this_inf2.config(loss=random_loss)
    '''


            
def main():
    print ("Use H1 as the centralized node")
    topo = SingleSwitchTopo(10)
    net = Mininet( topo=topo, link=TCLink )
    net.start()
    commandprompt = threading.Thread(target=start_CLI, args=(net,))
    commandprompt.start()
    time.sleep(5)
    print ('\n')


    print ('Starting varying packet loss')
    linkfunc(net)
    
    startslice = 0 #(0-29)
    
    for i in range (2,11):
        if i % 3 == 2:
            print ("host",i)
            endslice = startslice +1
            runcommand(i,startslice,endslice,net)
            print ("startingslice",startslice)
            print ("endingslice",endslice)
            startslice = endslice+1
            
        elif i %3 ==0:
            print ("host",i)
            endslice = startslice +2
            runcommand(i,startslice,endslice,net)
            print ("startingslice",startslice)
            print ("endingslice",endslice)
            startslice = endslice+1
            
        elif i %3 ==1:
            print ("host",i)
            endslice = startslice +4
            print ("startingslice",startslice)
            print ("endingslice",endslice)
            runcommand(i,startslice,endslice,net)
            startslice = endslice+1
        
    
    commandprompt.join()
            
if __name__ == '__main__':
    main()
