# Park!
<img src="park_demo.gif" alt="" />
<h2>Introduction</h2>
<p>As interns at NASA Langley, aka "The Sherpas", we worked on a lot of projects. Some of them involved machine learning, and, yes, this included a parking lot project with Georgia Tech. Due to architecture, security requirements, cross-lot tracking, etc., the NASA application was a bit complex, but here is a simplified version. It includes the full stack; SQL, Python, Linux scripts, HTML, CSS, JavaScript, and PHP, as well as cool computer vision and machine learning demos.</p>
<p>While the orginal system ran on three separate Red Hat Linux servers using Apache and MySQL, our demonstration system will run on the following system:</p>
<ul>
    <li>Oracle VirtualBox Virtual Machine (VM) Manager(using 6.0.6)</li>
    <li>CentOS Linux 7 (using 3.10.0-957.e17.x86_84)</li>
    <li>cron Time-Based Job Scheduler (using cronie-1.4.11-19.e17.x86_64)</li>
    <li>SQLite Relational Database Management System (RDBMS) (using 3.7.17)</li>
    <li>Extra Packages for Enterprise Linux (EPEL) (using epel-release-7-11.noarch)</li>
    <li>Nginx Hypertext Transfer Protocol (HTTP) Web Server (using nginx.x86_64 1:1.16.1-1.e17)</li>
    <li>Python 3 Programming Language Interpreter (using version 3.6.8)</li>
    <li>PHP Hypertext Preprocessor (using version 5.4.16) with Zend Engine Interpreter (using version 2.4.0)</li>
</ul>
<h2>Setup</h2>
<p>As we just said, we'll be using CentOS Linux 7 in VirtualBox for this demo, but you can use another virtual machine or an actual server if you like. Just make sure that your system has at least 2GB of memory; 16GB of hard disk space; 128MB of video memory; and a connection to the Internet. To start things off right, let's create a super user to avoid using the root user:</p>
<pre>
[root@localhost ~]# adduser park
[root@localhost ~]# passwd park
Changing password for user park.
New password: ********
Retype new password: ********
passwd: all authentication tokens updated successfully.
[root@localhost ~]# gpasswd -a park wheel
Adding user park to group wheel
[root@localhost ~]# su - park
[park@localhost ~]# 
</pre>
<p>Next, update the system using the following command:</p>
<pre>[park@localhost ~]# sudo yum update</pre>
<p>This may take a while, especially on a new system.</p>
<p>Once the system update is completed, make sure that we have everything we need:</p>
<ol>
    <li><b>cron Time-Based Job Scheduler</b> - cron should be already installed, but check anyway:
<pre>
[park@localhost ~]# whereis -b crontab | cut -d' ' -f2 | xargs rpm -qf
cronie-1.4.11-19.e17.x86_64
</pre>
        <p>If cron is not installed, install it using the following command:</p>
<pre>[park@localhost ~]# yum install cronie</pre>
    </li>
    <li><b>SQLite RDBMS</b> - SQLite should also be installed, but check anyway:
<pre>
[park@localhost ~]# sqlite3 -version
3.7.17 2013-05-20 00:56:22 118a3b35693b134d56ebd780123b7fd6f1497668
</pre>
        <p>If SQLite is not installed, install it using the following command:</p>
<pre>
[park@localhost ~]# sudo yum install sqlite
</pre>
    </li>
    <li><b>Nginx HTTP Web Server</b> - To use Nginx, install the <a href="https://fedoraproject.org/wiki/EPEL" target="_blank">Extra Packages for Enterprise Linux (EPEL)</a> first:
<pre>
[park@localhost ~]# sudo yum install epel-release
</pre>
        <p>Once the installation is completed, install Nginx:</p>
<pre>
[park@localhost ~]# sudo yum install nginx
</pre>
        <p>Once Nginx is installed, start the server. In addition, run the second command so that Nginx automatically starts when the system boots up:</p>
<pre>
[park@localhost ~]# sudo systemctl start nginx
[park@localhost ~]# sudo systemctl enable nginx
Created symlink from /etc/systemd/system/multi-user.target.wants/nginx.service to /usr/lib/systemd/system/nginx.service.
</pre>
        <p>Once the server is started, we have several ways to access the web server:</p>
        <ul>
            <li>Method #1: Open a browser in the VM - This is the easiest way, but we won't be able to access the site remotely. Install and start a desktop GUI, such as GNOME, using the following commands:
<pre>
sudo yum groupinstall "GNOME Desktop"
startx
</pre>
            <p>Once the desktop appears, open a browser and navigate to localhost (http://127.0.0.1) and the Welcome page should appear:</p>
            <img src="images/readme01.png" alt="" style="width:640px;" />
            </li>
            <li>Method #2: Access the web server through the VM host's browser - This method allows us to "test" the web server over a an actual connection, even though everything occurs on the VM's host computer.
            <p>First, shutdown the VM and access the network settings:</p>
            <img src="images/readme02.png" alt="" style="width:640px;" />
            <p>Click on <b>Port Forwarding</b>. Set the Host Port to 8080 and the Guest Port to 80; click on <b>OK</b> when you are done:</p>
            <img src="images/readme03.png" alt="" style="width:640px;" />
            <p>Restart the VM and enter the following commands:</p>
<pre>
sudo firewall-cmd --zone=public --add-service=http --permanent
sudo firewall-cmd --reload
</pre>
            <p>Open a browser on the host machine, navigate to http://127.0.0.1:8080, and the Welcome page should appear:</p>
            <img src="images/readme05.png" alt="" style="width:640px;" />
            </li>
            <li>Method #3: If you are using an actual server, you can access it by entering its public IP address. Pull up the network interfaces using the following command:
<pre>ip addr</pre>
            <p>The first network interface name should be localhost, while the following name, which has the attributes &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt;, should be the name of the public address. Navigate to the INET IP address in a browser, and the Welcome page should appear.</p>
            </li>
        </ul>
    </li>
    <li><b>PHP Hypertext Preprocessor</b> - We'll be using PHP as the intermediary between the front and back ends. To install PHP, run the following command:
<pre>[park@localhost ~]# sudo yum install php</pre>
    </li>
    <li><b>Python 3 Programming Language Interpreter</b> - While Python 2 is installed with CentOS by default, we will need Python 3 to run our computer vision and machine learning scripts, specically Python 3.6.x. To install Python, run the following command:
<pre>
[park@localhost ~]# sudo yum install centos-release-scl
[park@localhost ~]# sudo yum install rh-python36
[park@localhost ~]# sudo yum groupinstall 'Development Tools'
</pre>
    </li>
    <li>Before continuing, do another update of the system using the following command:
<pre>[park@localhost ~]# sudo yum update</pre>
    </li>
</ol>
<p>Just in case, we'll double check everything is installed and updated using the following commands:</p>
<pre>
[park@localhost ~]# whereis -b crontab | cut -d' ' -f2 | xargs rpm -qf
[park@localhost ~]# sqlite3 -version
[park@localhost ~]# nginx -v
[park@localhost ~]# php -v
[park@localhost ~]# scl enable rh-python36 bash
[park@localhost ~]# python ––version
</pre>
<img src="images/readme06.png" alt="" style="text-align: center; width: 640px;" />
<p>Please note that, in order to use Python 3.6.x, we will have to to launch a new shell instance using the Software Collection scl tool each time we start the system. To have the system do this automatically when loading, we'll add a bashrc file to our profile using the following commands:</p>
<pre>
[park@localhost ~]# cd etc/profile.d
[park@localhost ~]# sudo nano bashrc
</pre>
<p>Once the file is open in nano, enter the following text:</p>
<pre>
# Make Python 3.6.x the default Python interpreter
scl enable rh-python36 bash
</pre>
<p>Hit [CTRL][O] to WriteOut and [CTRL][X] to Exit. Now, return to the park direcctory:</p>
<pre>
[park@localhost ~]# cd /home/park
</pre>
<p>Whew! That was a lot of setting up! Now let's get to coding...</p> 
<h2>The Data Model</h2>
<h2>The Scripts<h2>
<h2>The Back-End</h2>
<h2>The Front-End</h2>
<h2>That's All Folks!</h2>