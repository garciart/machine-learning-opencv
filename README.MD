# Park!
<div style="text-align: center;"><img src="README_images/park_demo.gif" alt="Welcome to Park!" /></div>
<h2>Introduction</h2>
<p>As interns at NASA Langley, aka "The Sherpas", we worked on a lot of projects. Some of them involved machine learning, and, yes, this included a parking lot project with Georgia Tech.</p>
<p>The use case was that finding parking is a problem for everyone. In 2016, drivers in New York City spent an average of 107 hours and $2243 a year looking for parking (INRIX, 2017). In addition, there are over 120 outstanding requests for proposals for city and institutional parking management (IPMI, 2019). NASA Langley, in an effort to reduce the stress of its employees, began looking into implementing its own parking management system.</p>
<p>The original concept for LaRC Park, proposed by a NASA employee, was to place sensors in each parking space. While this solution may have provided the most accurate results, we realized that due to exposure to weather, security, gaps in connectivity, etc., individual sensors would be impractical. We then looked at different systems, including Light Detection and Ranging (LIDAR) and image recognition.</p>
<p>Each approach had its own issues; LIDAR systems required power and connectivity where there was none, and most image recognition systems required a high, almost overhead, angle of view.</p>
<p>The break came after reading <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400" target="_blank" title="Snagging parking spaces with Mask R-CNN and Python">Adam Geitgey's awesome article on object detection</a>, which introduced us to <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="Matterport Mask R-CNN">Matterport Mask R-CNN</a> (Geitgey, 2019). We realized that this implementation of Mask R-CNN was something we could use with the resources available to us, especially the lower and more practical viewing angle. However, we also realized that we could not use Adam's code as is.</p>
<p>First, TensorFlow, the heart of the system, is expensive in terms of processing power; even when using his solution on a single system for a single user looking at single parking lot, we could hear the fans of our computers going into overdrive. Satisfying multiple instantiations by multiple users would quickly overwhelm any of our available systems.</p>
<p>Second, to prepare TensorFlow, the system needs to process a training and test set. This takes a few, but noticeable, number of seconds; check out another project, <a href="http://acronymsfortina.rgprogramming.com/index.html" target="_blank" title="Acronyms for Tina">Acronyms for Tina</a>, to experience this lag for yourself. In addition, the system had to observe several frames before making a prediction, adding to total processing time. Finally, the system had to process two sets of frames in order to make a prediction. The first set of frames had to be a full lot, which would be compared to images of the current state of the lot, to determine if there were any spaces available. </p>
<p>The combined delay meant that user could wait up to a minute before the system provided them with an optimal parking spot. This delay would only increase as lots were added to the system.</p>
<p>Therefore, we took the system back to the drawing board and came up with a different solution; we would create zones and use Mask R-CNN to count how many cars were in each zone. In addition, instead of users running TensorFLow on demand, the system would:</p>
<ol>
<li>Use the cron task scheduler to run a Python script that would process each lot on an optimized schedule.</li>
<li>Collect the results in a database.</li>
<li>Have the front end pull data from the database.</li>
</ol>
<p>This would prevent the user from experiencing the processing lag. In addition, while the system still needed time to process the frames. it was much faster than the original system.</p>
<p>Due to architecture, security requirements, cross-lot tracking, etc., the NASA application was a bit complex, but here are a series of demos, written in Python, that breakdown the way the system works. While the original system ran on three separate Red Hat Linux servers using Apache and MySQL, our demonstration system will run on the following system:</p>
<ul>
	<li>Oracle VirtualBox Virtual Machine (VM) Manager(using 6.0.6)</li>
	<li>CentOS Linux 7 (using 3.10.0-957.e17.x86_84)</li>
	<li>cron Time-Based Job Scheduler (using cronie-1.4.11-19.e17.x86_64)</li>
	<li>SQLite Relational Database Management System (RDBMS) (using 3.7.17)</li>
	<li>GNOME Desktop Version 3.28.2
	<li>Python 3 Programming Language Interpreter (using version 3.6.8)</li>
	<li>Extra Packages for Enterprise Linux (EPEL) (using epel-release-7-11.noarch)</li>
	<li>Nginx Hypertext Transfer Protocol (HTTP) Web Server (using nginx.x86_64 1:1.16.1-1.e17)</li>
	<li>PHP Hypertext Preprocessor (using version 5.4.16) with Zend Engine Interpreter (using version 2.4.0)</li>
</ul>
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