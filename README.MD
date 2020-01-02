# Park!
<div style="text-align: center;"><img src="README_images/park_demo.gif" alt="Welcome to Park!" /></div>
<h2>Introduction</h2>
<p>As interns at NASA Langley, aka "The Sherpas", we worked on a lot of projects. Some of them involved machine learning, and, yes, this included a parking lot project.</p>
<p>The use case was that finding parking is a problem for everyone. In 2016, drivers in New York City spent an average of 107 hours and $2243 a year looking for parking (INRIX, 2017). In addition, there are over 120 outstanding requests for proposals for city and institutional parking management (IPMI, 2019). NASA Langley, in an effort to reduce its own parking problems, began looking into implementing a parking management system.</p>
<p>We looked at various solutions, including sensors in each parking space; Light Detection and Ranging (LIDAR) to count cars entering and exiting parking areas; etc. Each approach had issues that made it impractical to implement with the resources we had; for example, most image recognition systems we looked at required a high, almost overhead, angle of view.</p>
<p>Finally, we came accross <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400" target="_blank" title="Snagging parking spaces with Mask R-CNN and Python">Adam Geitgey's awesome article on object detection</a>, which introduced us to <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="Matterport Mask R-CNN">Matterport Mask R-CNN</a> (Geitgey, 2019). We realized that his implementation of Mask R-CNN was something we could use with the resources available to us, especially the lower and more practical viewing angle. However, Adam’s solution, designed for a single system for a single user looking at a single parking lot, was too expensive in terms of processing power and time. The first time we implemented it, it sent our cooling fans into overdrive and hung our computers for several minutes; run another of our projects, <a href="http://acronymsfortina.rgprogramming.com/index.html" target="_blank" title="Acronyms for Tina">Acronyms for Tina</a>, on localhost for a similar, albeit less intense, experience.</p>
<p>Still, it worked, so using what we learned, we went back to the drawing board to develop a system that would rotate through multiple lots; analyze and collect parking data, and present this information to users in a timely manner. After much trial and error, we came up with a system that leveraged Linux scripting, Python, OpenCV, TensorFlow, SQL, and PHP to:</p>
<ol>
<li>Use a cron task scheduler to access the cameras overlooking each lot in turn.</li>
<li>Capture a frame from each camera; divide it into zones (e.g., employee, handicap, etc.), and count the number of vehicles in each zone.</li>
<li>Aggregate the results and collect them in a database.</li>
<li>Have the front-end pull data for users from the database.</li>
</ol>
<p>Due to architecture, security requirements, cross-lot tracking, etc., the NASA application was a bit complex, but here are a series of demos, written in Python, that breakdown the way the system works. Besides a practical application of TensorFlow, Mask R-CNN, and computer vision, this is also a very good introduction to the Common Objects in COntext (COCO) dataset. Have fun and good luck!</p>
<hr>
<h2>Setup</h2>
<p>As we just said, we'll be using CentOS Linux 7 in VirtualBox for this demo, but you can use another virtual machine or an actual server if you like. Just make sure that your system has at least 2GB of memory; 16GB of hard disk space; 128MB of video memory; and a connection to the Internet.</p>
<p style="border:1px solid; padding: 6px; text-align: center;"><b>NOTE</b> - If you do choose to use CentOS, we've included another README file name CENTOS with directions on how to do so.</p>
<p style="border:1px solid; padding: 6px; text-align: center;"><b>NOTE</b> - To view the images from the demo scripts, you will need an X Server. We used the GNOME Graphical User Interface (GUI) for development; it came with a display server; it was lightweight; and it allowed us to cut-and-paste from the host machine into the VM's Terminal.</p>
<hr>
<div style="width: 100%; text-align: center;">
<img src="sherpa_q2a_logo.png" alt="Sherpa Logo" style="height:240px;" />
</div>
<p>The Sherpas are Jrei Dimanno, Justine Forrest, Rob Garcia, John Graham, Tanner Griffin, Ryan Hashi, Brandon Hutton, Gabriel Jacobs, Fernando Jauregui, Wesley Madden, and Doug Trent</p>
<hr>
<p>Geitgey, A. (2019, January 21). Snagging parking spaces with Mask R-CNN and Python. Retrieved from <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400</p>" target="_blank" title="Snagging parking spaces with Mask R-CNN and Python">https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400</p></a>
<p>INRIX. (2017, July 12). Searching for parking costs Americans $73 billion a year. Retrieved from <a href="http://inrix.com/press-releases/parking-pain-us/</p>" target="_blank" title="Searching for parking costs Americans $73 billion a year">http://inrix.com/press-releases/parking-pain-us/</p></a>
<p>International Parking and Mobility Institute. (2015, October 12). Open requests for proposals. Retrieved July 31, 2019, from <a href="https://www.parking.org/membership/member-resources/rfps/</p>" target="_blank" title="Open requests for proposals">https://www.parking.org/membership/member-resources/rfps/</p></a>
<p>Matterport, Inc. (2019, March 10). matterport/Mask_RCNN. Retrieved August 2, 2019, from <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="matterport/Mask_RCNN">https://github.com/matterport/Mask_RCNN</a></p>