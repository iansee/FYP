#!/usr/bin/python
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/mininet-2.2.2-py2.7.egg')
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
            if n == 4:
                # 10 Mbps, 5ms delay, 2% loss
                self.addLink( host, switch, bw=10, delay='5ms',loss=2)
            else:
                self.addLink( host, switch, bw=100)

def perfTest():
    "Create network and run simple performance test"
    topo = SingleSwitchTopo( n=4 )
    net = Mininet( topo=topo, link=TCLink )
    net.start()
    CLI(net)
    
    

if __name__ == '__main__':
    perfTest()
