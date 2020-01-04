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
<p>This will create a folder "Park" with all the code in the right place. If you are using a shared folder, fetch into your shared folder instead of cloning:</p>
<pre>
[park@localhost ~]# cd Park # replace Park with the name of your shared folder
[park@localhost Park]$ git init
[park@localhost Park]$ git remote add origin https://github.com/garciart/Park.git
[park@localhost Park]$ git fetch
[park@localhost Park]$ git checkout origin/master -ft
</pre>
<p>As a rule, we don't use the "Master" branch, just like we don't use "root" for development. You don't have to, but we suggest creating a separate branch:</p>
<pre>
[park@localhost ~]# cd Park # replace Park with the name of your shared folder
[park@localhost Park]$ git branch park # replace park with your username
[park@localhost Park]$ git checkout park # replace park with your username
[park@localhost Park]$ git status
</pre>
<p>In addition, remember to use "Park" as your development folder, not the home folder (in our case, "park").</p>
</div>
<br>
<div style="text-align: center;"><img src="README_images/readme02.png" alt="Cloning the Park repository" style="height: 400px;" /></div>
<p>Clone the excellent <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="Mask R-CNN engine from Matterport">Mask R-CNN engine from Matterport</a>, which will serve as our object detection and instance segmentation engine using the following commands:</p>
<pre>
[park@localhost Park]$ git clone https://github.com/matterport/Mask_RCNN.git
[park@localhost Park]$ cd Mask_RCNN
[park@localhost Mask_RCNN]$ ls
[park@localhost Mask_RCNN]$ cat requirements.txt
</pre>
<br>
<div style="text-align: center;"><img src="README_images/readme03.png" alt="Cloning the Park repository" style="height: 400px;" /></div>
<p>Note the setup.py file, which will build and install the Mask RCNN engine, and the requirements.txt file, which contains the list of modules and packages we will need to run the Mask RCNN engine:</p>
<ul>
<li><a style="color: orange;" href="https://numpy.org/" title="numpy" target="_blank">numpy</a> - support for high-level mathematics involving arrays and matrices.</li>
<li><a style="color: orange;" href="https://scipy.org/scipylib" title="scipy" target="_blank">scipy</a> - support for scientific and technical computing.</li>
<li><a style="color: orange;" href="https://python-pillow.org/" title="Pillow" target="_blank">Pillow</a> - Python Imaging Library to open, manipulate and save image files.</li>
<li><a style="color: orange;" href="https://cython.org/" title="cython" target="_blank">cython</a> - support for C extensions and datatypes in Python.</li>
<li><a style="color: orange;" href="https://matplotlib.org/" title="matplotlib" target="_blank">matplotlib</a> - support for 2d plotting.</li>
<li><a style="color: orange;" href="https://scikit-image.org/" title="scikit-image" target="_blank">scikit-image</a> - support for image processing.</li>
<li><a style="color: orange;" href="https://www.tensorflow.org/" title="tensorflow" target="_blank">tensorflow>=1.3.0</a> - support for machine learning and tensor mathematics.</li>
<li><a style="color: orange;" href="https://keras.io/" title="keras" target="_blank">keras>=2.0.8</a> - support for neural networks.</li>
<li><a style="color: orange;" href="https://opencv.org/" title="opencv" target="_blank">opencv-python</a> - support for real-time computer vision.</li>
<li><a style="color: orange;" href="https://www.h5py.org/" title="h5py" target="_blank">h5py</a> - support for implementing HDF5 binary data format.</li>
<li><a style="color: orange;" href="https://imgaug.readthedocs.io/en/latest/" title="imgaug" target="_blank">imgaug</a> - support for for image augmentation in machine learning.</li>
<li><a style="color: orange;" href="https://ipython.org/" title="IPython" target="_blank">IPython[all]</a> - support for interactive computing.</li>
</ul>
<p>However, before executing installing requirements.txt, we have to make some changes. Unfortunately, Matterport's version of Mask R-CNN cannot use TensorFlow 2.0 or Keras 2.3 or above. In addition, we will need to install <a href="https://docs.opencv.org/master/" title="opencv-python-contrib" target="_blank">OpenCV's extra modules</a> to display images. Using your favorite editor, change the TensorFlow and Keras lines to the following:</p>
<pre>
tensorflow&gt;=1.3.0,&lt;2.0
keras&gt;=2.0.8,&lt;2.3
</pre>
<p>In addition, add the following line after the OpenCV line:</p>
<pre>
opencv-contrib-python
</pre>
<div style="text-align: center;"><img src="README_images/readme04.png" alt="Editing requirements.txt" style="height: 400px;" /></div>
<p>We also have to make sure our <a style="color: orange;" href="https://setuptools.readthedocs.io/en/latest/" title="Setuptools" target="_blank">Setuptools</a> are up-to-date. Otherwise, we may run into errors extracting and creating modules and packages from requirements.txt. We have to make sure <a style="color: orange;" href="https://docs.python.org/3/library/tkinter.html" title="tkinter" target="_blank">tkinter, the standard Python interface to the Tk GUI toolkit</a>, is installed as well, or you may get a "matplotlib is currently using a non-GUI backend" error. Finally, we Therefore, enter the following commands:</p>
<pre>
[park@localhost Mask_RCNN]$ pip3 install --upgrade setuptools --user
[park@localhost Mask_RCNN]$ sudo yum -y install python3-tkinter
</pre>
<p>Once the installation is complete, install Mask R-CNN's requirements using the following command:
<pre>
[park@localhost Mask_RCNN]$ pip3 install -r requirements.txt --user
</pre>
<div style="text-align: center;"><img src="README_images/readme05.png" alt="Installing Mask R-CNN requirements" style="height: 400px;" /></div>
<p>Hopefully, everything went well; if not, check the verbose comments for any errors and correct them. Remember, you may have to use a different flavor of a package depending on the system you are using. For example, we used Red Hat Linux, so for some packages, we had to use their version (e.g., python36-mysql, etc.). You can look for packages using:
<pre>
[park@localhost Mask_RCNN]$ pip3 list
- or -
[park@localhost Mask_RCNN]$ yum search [package name]
</pre>
<p>Once everything is set, run the setup script using the command:</p>
<pre>
[park@localhost Mask_RCNN]$ python3 setup.py install --user
</pre>
<div style="text-align: center;"><img src="README_images/readme06.png" alt="Setting up Mask R-CNN" style="height: 400px;" /></div>
<p>Once again, check the verbose comments for any errors and correct them. Finally, go back one directory and download the Common Objects in Context (COCO) dataset by using the following commands:</p>
<pre>
[park@localhost Mask_RCNN]$ cd ..
[park@localhost Park]$ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5</pre>
<p>The COCO dataset consists of over 100,000 images of 80 objects, including cars (#3), buses (#6), and trucks (#8); find out more at <a style="color: orange;" href="http://cocodataset.org" title="COCO Dataset" target="_blank">http://cocodataset.org</a>. Please note that you can replace this dataset with a more specific set for vehicle detection, but for now, we will use the COCO dataset:</p>
<div style="text-align: center;"><img src="README_images/readme07.png" alt="Getting the COCO dataset" style="height: 400px;" /></div>
<h2>Running the Scripts</h2>
<p>Now comes the fun part: Our Python code is a combination of Matterport's open-source Mask-RCNN's samples (Matterport, 2019), Alex Geitey's excellent article on detecting open parking spaces (Geitey, 2019), and our own embellishments (all MIT licensed, of course). By the time you are done working through these demos, you should have a good understanding how the heavy lifting occurs on the back end.</p>
<div style="border:1px solid; padding: 6px;">
<p><b>NOTE</b> - To view these images generated by the demo scripts, you will need an X Server; otherwise, OpenCV will display a "Failed to Start the X server" error. Our solution was to use the GNOME Graphical User Interface (GUI) for development; it came with a display server; it was lightweight; and it allowed us to cut-and-paste from the host machine into the VM's Terminal. To do so on Linux, use the following commands:</p>
<pre>
sudo yum -y groupinstall "GNOME Desktop"
sudo startx
</pre>
</div>
<p>Download the demos to your development folder if you did not clone or fetch the repo. There are six demos, and each one adds new functionality to Park.</p>
<ul>
    <li>park_demo1.py - OpenCV image capture and display demo.</li>
    <li>park_demo2.py - Adds Mask R-CNN object detection to the demo.</li>
    <li>park_demo3.py - Adds recognizing and counting vehicles to the demo.</li>
    <li>park_demo4.py - Adds creating and displaying zones to the demo.</li>
    <li>park_demo5.py - Adds counting vehicles in zones to the demo.</li>
    <li>park_demo6.py - El Super Demo: Incorporates communicating with the database and the functionality of all of the previous demos.</li>
</ul>
<p>You will also need the log, capture, image, and video folders. If you cloned or fetched the repository, your directory should look like this:</p>
<div style="text-align: center;"><img src="README_images/readme08.png" alt="Development folder" style="height: 400px;" /></div>
<p>Once you are all set up, open the code for each demo in your favorite editor as we get to it and follow along:</p>
<hr>
<h3>Demo 1 - OpenCV Image Capture and Display</h3>
<p>This one is pretty easy, but using OpenCV is the foundation of Park's image recognition system. Note that we listed three FRAME_SOURCE's (Lines 11-19), with two of them commented out:</p>
<pre style="white-space: pre-wrap; word-break: keep-all; background-color: black; border: 1px solid white; padding: 12px;">
# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos

\# Image, video or camera to process - set this to 0 to use your webcam instead of a file
\# FRAME_SOURCE = [(IMAGE_DIR + "/demo_image1.jpg"),(IMAGE_DIR + "/demo_image2.jpg"),(IMAGE_DIR + "/demo_image3.jpg")]
\# FRAME_SOURCE = [(VIDEO_DIR + "/demo_video1.mp4"),(VIDEO_DIR + "/demo_video2.mp4"),(VIDEO_DIR + "/demo_video3.mp4")]
\# FRAME_SOURCE = [(IMAGE_DIR + "/demo_image.jpg")]
FRAME_SOURCE = ["https://raw.githubusercontent.com/garciart/Park/master/demo_images/demo_image.jpg"]
</pre>
<p>Park can work with images, videos and live streams. While we use demo_image.jpg, feel free to switch sources during the first three demos. However, to work correctly with this tutorial, Demos 4 through 5 need to use demo_image.jpg, while Demo 6 pulls what it needs from the database you will create later.</p>
<p>Note the colors. OpenCV uses a Blue-Green-Red (BGR) color model, which we converted to an RGB color model for Mask R-CNN (Line 33):</p>
<pre style="white-space: pre-wrap; word-break: keep-all; background-color: black; border: 1px solid white; padding: 12px;">
# Convert the image from BGR color (which OpenCV uses) to RGB color
rgb_image = frame[:, :, ::-1]</pre>
<p>We did not notice a drop in detection accuracy, but we do what the great guys at Matterport tell us to do!</p>
<p>IMAGE GOES HERE</p>
<div style="border:1px solid; padding: 6px;">
<p><b>NOTE</b> - After some installations, you may receive the following error when attempting to use cv2.imshow:</p>
<pre>
The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support.
</pre>
<p>To correct this problem, force reinstall OpenCV using the following commands:</p>
<pre>
pip3 install --upgrade --force-reinstall opencv-python --user
pip3 install --upgrade --force-reinstall opencv-contrib-python --user
</pre>
</div>
<hr>

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