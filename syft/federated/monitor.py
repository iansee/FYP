import psutil

class monitoring():
    def __init__(self):
        self.bytes_sent = None
        self.bytes_recv = None
        self.packets_sent = None
        self.packets_recv = None
        self.err_in = None
        self.err_out = None
        self.drop_in = None
        self.drop_out = None

        interfaces = psutil.net_io_counters(pernic=True)
        for interface in interfaces:
            if 'eth0' in interface:
                self.my_interface = interface
            
        
    def start(self):
        interfaces = psutil.net_io_counters(pernic=True)
        network_obj = interfaces[self.my_interface]
        self.bytes_sent = network_obj.bytes_sent
        self.bytes_recv= network_obj.bytes_recv
        self.packets_sent=network_obj.packets_sent
        self.packets_recv=network_obj.packets_recv
        self.err_in=network_obj.errin
        self.err_out=network_obj.errout
        self.drop_in=network_obj.dropin
        self.drop_out=network_obj.dropout 
                
    def stop(self):
        interfaces = psutil.net_io_counters(pernic=True)
        network_obj = interfaces[self.my_interface]

        new_bytes_sent = network_obj.bytes_sent - self.bytes_sent
        new_bytes_recv = network_obj.bytes_recv - self.bytes_recv
        new_packets_sent =network_obj.packets_sent - self.packets_sent
        new_packets_recv =network_obj.packets_recv - self.packets_recv
        new_err_in =network_obj.errin - self.err_in
        new_err_out =network_obj.errout - self.err_out
        new_drop_in =network_obj.dropin - self.drop_in
        new_drop_out = network_obj.dropout - self.drop_out
        
        return {"bytes sent" : new_bytes_sent, "bytes recieved" :new_bytes_recv,
                "packets sent" : new_packets_sent , "packets recieved": new_packets_recv,
                "Incoming errors" : new_err_in, "Outgoing errors" : new_err_out,
                "Drop ins": new_drop_in, "Drop outs" : new_drop_out}
                
