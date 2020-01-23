Installation:

Pre-Req =>
Xming
Putty
Mininet x86_64 mn-trusty64server-170321-14-17-08

Mininet installation:
1.	Image Setup
-Medium management in Oracle Virtual Box
-Adapter2: Host only adapter for Xming
	-/etc/network/interfaces
		-add eth1
2.	Reformatting of /dev/sda to extend blocks of /dev/sda1
-	Sudo fdisk /dev/sda
-	Delete all partitions
-	Create partition 1
-	Save and reboot
-	Sudo Resize2fs /dev/sda1
3.	Sudo apt-get update
4.	Installation of Ananconda:
-	Sudo apt-get install curl
-	Curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
-	bash Anaconda3-2019.03-Linux-x86_64.sh
-	source ~/.bashrc
5.	Installation of pytorch - installation of newest version of pytorch such that it meets the requirements of pysyft
-	Conda install pytorch==1.1.0 torchvision cudatoolkit=10.0 -c pytorch
6.	Installation of tensorflow
-	Conda install -c conda-forge tensorflow
7.	Installation of PySyft
-	Pip install syft
8.	https://github.com/OpenMined/PySyft/issues/2275 - downgrade pytorch
-	Sudo conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch
9.	Installation of updated tensorflow
-	Sudo pip install -U "tensorflow=1.*"
10.	Enable mininet by adding the file path
	sudo echo ~/.bashrc export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python2.7/dist-packages/mininet-2.2.2-py2.7.egg"
	
*Extra samba:
Apt-get install samba
Sudo nano /etc/samba/smb.conf
[share]
Path = /
Read only = no
Browsable  = yes

Sudo service smbd restart
Sudo ufw allow samba
Sudo smbpasswd – a mininet


*Using the tensorflow federated dataset:
To circumvent the usage of tff, extract the HDFClient data file which is the only relevant class we need from tff
This will resolve the conflicts between tensorflow version required for syft and tensorflow
