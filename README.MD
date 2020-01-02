# Park!
<div style="text-align: center;"><img src="README_images/park_demo.gif" alt="Welcome to Park!" /></div>
<h2>Introduction</h2>
<p>As interns at NASA Langley, aka "The Sherpas", we worked on a lot of projects. Some of them involved machine learning, and, yes, this included a parking lot project.</p>
<p>The use case was that finding parking is a problem for everyone. In 2016, drivers in New York City spent an average of 107 hours and $2243 a year looking for parking (INRIX, 2017). In addition, there are over 120 outstanding requests for proposals for city and institutional parking management (IPMI, 2019). One of the original Sherpas, who had been working on a similar project at Georgia Tech, suggested creating a parking management system for NASA Langley's Digital Transformation initiative, and it was approved.</p>
<p>We looked at various solutions, including sensors in each parking space; Light Detection and Ranging (LIDAR) to count cars entering and exiting parking areas; etc. Each approach had issues that made it impractical to implement with the resources we had; for example, most image recognition systems we looked at required a high, almost overhead, angle of view.</p>
<p>Finally, we came accross <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400" target="_blank" title="Snagging parking spaces with Mask R-CNN and Python">Adam Geitgey's awesome article on object detection</a>, which introduced us to <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="Matterport Mask R-CNN">Matterport Mask R-CNN</a> (Geitgey, 2019). We realized that his implementation of Mask R-CNN was something we could use with the resources available to us, especially the lower and more practical viewing angle. However, Adam’s solution, designed for a single system for a single user looking at a single parking lot, was too expensive in terms of processing power and time. The first time we implemented it, it sent our cooling fans into overdrive and hung our computers for several minutes; run another of our projects, <a href="http://acronymsfortina.rgprogramming.com/index.html" target="_blank" title="Acronyms for Tina">Acronyms for Tina</a>, on localhost for a similar, albeit less intense, experience.</p>
<p>Still, it worked, so using what we learned, we went back to the drawing board to develop a system that would rotate through multiple lots; analyze and collect parking data, and present this information to users in a timely manner. After much trial and error, we came up with a system that leveraged Linux scripting, Python, OpenCV, TensorFlow, SQL, and PHP to:</p>
<ol>
<li>Use a cron task scheduler to access the cameras overlooking each lot in turn.</li>
<li>Capture a frame from each camera; divide it into zones (e.g., employee, handicap, etc.), and count the number of vehicles in each zone.</li>
<li>Aggregate the results and collect them in a database.</li>
<li>Have the front-end pull data for users from the database.</li>
</ol>
<p>Due to architecture, security requirements, cross-lot tracking, etc., the NASA application was a bit complex, and for those smae reasons, we will not recreate it here. However, here are a series of demos, written in Python, that breakdown the way the machine learning aspect works. Besides a practical application of TensorFlow, Mask R-CNN, and computer vision, this is also a very good introduction to the Common Objects in COntext (COCO) dataset. Have fun and good luck!</p>
<hr>
<h2>Setting Up the Development Environment</h2>
<p>As we just said, we'll be using CentOS Linux 7 in VirtualBox for this demo, but you can use another virtual machine or an actual server if you like. Just make sure that your system has at least 2GB of memory; 16GB of hard disk space; 128MB of video memory; and a connection to the Internet.</p>
<p style="border:1px solid; padding: 6px;"><b>NOTE</b> - If you do choose to use CentOS, we've included another README file name CENTOS with directions on how to set up your environment.</p>
<div style="border:1px solid; padding: 6px;">
<p><b>NOTE</b> - While the production code only generated data, the demo scripts also display images. To view these images, you will need an X Server. Our solution was to use the GNOME Graphical User Interface (GUI) for development; it came with a display server; it was lightweight; and it allowed us to cut-and-paste from the host machine into the VM's Terminal. To do so on Linux, use the following commands:</p>
<pre>
sudo yum -y groupinstall "GNOME Desktop"
sudo startx
</pre>
</div>
<p>Our first step is to make sure the operating system is up to date. On CentOS, we would use the following command; other flavors of Linux may use "sudo apt-get update" instead, while Windows users can run "wuauclt.exe /updatenow":</p>
<pre>
[park@localhost ~]# sudo yum -y update
</pre>
<p>This may take a while, especially on a new system.</p>
<p>Once the system update is completed, make sure that the tools needed for development are installed:</p>
<ol>
<li><b>Python 3 Programming Language Interpreter and PIP Python Package Installer</b> - While Python 2 is installed with CentOS by default, we will need Python 3 to run our computer vision and machine learning scripts, specically Python 3.6.x. There are a few ways of doing this, but we will use the IUS Community Repo; for an in-depth look at options, check out <a href="https://www.hogarthuk.com/?q=node/15" title="Running newer applications on CentOS" target="_blank">this link from James Hogarth</a>. To install Python, run the following command:
<pre>
[park@localhost ~]# sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
[park@localhost ~]# sudo yum -y install python36u
[park@localhost ~]# sudo yum -y install python36u-pip
[park@localhost ~]# sudo yum -y install python36u-devel
[park@localhost ~]# python3 --version
[park@localhost ~]# pip3 --version
</pre>
</li>
<li><b>SQLite RDBMS</b> - For portability (and to keep this README as short as possible), we'll be using SQLite. Check the version using the following command:
<pre>
[park@localhost ~]# sqlite3 -version
3.7.17 2013-05-20 00:56:22 118a3b35693b134d56ebd780123b7fd6f1497668
</pre>
<p>If SQLite is not installed, install it using the following command:</p>
<pre>
[park@localhost ~]# sudo yum -y install sqlite
</pre>
</li>
<li><b>cron Time-Based Job Scheduler</b> - cron should already be installed by default, but check anyway:
<pre>
[park@localhost ~]# whereis -b crontab | cut -d' ' -f2 | xargs rpm -qf
cronie-1.4.11-19.e17.x86_64
</pre>
<p>If cron is not installed, install it using the following command:</p>
<pre>[park@localhost ~]# yum -y install cronie</pre>
</li>
</ol>
<p>Alright! Before continuing, let's do another update of the system using the following command:</p>
<pre>[park@localhost ~]# sudo yum -y update</pre>
<p>Just in case, we'll double check everything is installed and updated using the following commands:</p>
<pre>
[park@localhost ~]# python3 ––version
[park@localhost ~]# pip3 ––version
[park@localhost ~]# sqlite3 -version
[park@localhost ~]# whereis -b crontab | cut -d' ' -f2 | xargs rpm -qf
</pre>
<div style="text-align: center;"><img src="README_images/readme01.png" alt="Verifying initial setup" style="height: 400px;" /></div>
<h2>Adding the Machine Learning Packages</h2>
<p>Our next step is to add the packages we will need to run the scripts.</p>
<div style="border:1px solid; padding: 6px;">
<p><b>NOTE</b> - If you like, you can clone this repository into your folder: <strong>Just make sure you do so before installing the ML packages.</strong> Use the following command to clone the repo:</p>
<pre>git clone https://github.com/garciart/Park.git</pre>
<p>This will create a folder "Park" with all the code in the right place:</p>
<pre>
[park@localhost ~]# cd Park # replace Park with the name of your shared folder
[park@localhost Park~]$ git init
[park@localhost Park~]$ git remote add origin https://github.com/garciart/Park.git
[park@localhost Park~]$ git fetch
[park@localhost Park~]$ git checkout origin/master -ft
[park@localhost Park~]$ git branch park # replace park with your username
[park@localhost Park~]$ git checkout park # replace park with your username
</pre>
<p>If you are using a shared folder, fetch into your shared folder instead of cloning. In addition, remember to use "Park" as your development folder, not the home folder (in our case, "park").</p>
</div>
<hr>
<div style="width: 100%; text-align: center;">
<img src="sherpa_logo.png" alt="Sherpa Logo" style="height:240px;" />
</div>
<p style="text-align: center;">The Sherpas are Jrei Dimanno, Justine Forrest, Rob Garcia, John Graham, Tanner Griffin, Ryan Hashi, Brandon Hutton, Gabriel Jacobs, Fernando Jauregui, Wesley Madden, and Doug Trent</p>
<hr>
<p>Geitgey, A. (2019, January 21). Snagging parking spaces with Mask R-CNN and Python. Retrieved from <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400</p>" target="_blank" title="Snagging parking spaces with Mask R-CNN and Python">https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400</p></a>
<p>INRIX. (2017, July 12). Searching for parking costs Americans $73 billion a year. Retrieved from <a href="http://inrix.com/press-releases/parking-pain-us/</p>" target="_blank" title="Searching for parking costs Americans $73 billion a year">http://inrix.com/press-releases/parking-pain-us/</p></a>
<p>International Parking and Mobility Institute. (2015, October 12). Open requests for proposals. Retrieved July 31, 2019, from <a href="https://www.parking.org/membership/member-resources/rfps/</p>" target="_blank" title="Open requests for proposals">https://www.parking.org/membership/member-resources/rfps/</p></a>
<p>Matterport, Inc. (2019, March 10). matterport/Mask_RCNN. Retrieved August 2, 2019, from <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="matterport/Mask_RCNN">https://github.com/matterport/Mask_RCNN</a></p>