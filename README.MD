# Park!
<div style="text-align: center;"><img src="README_images/park_demo.gif" alt="Welcome to Park!" /></div>
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
<hr>
<p>Parking is a problem for everyone, not just NASA. Some research estimates that New York City drivers spend 107 hours $2243 a year looking for parking (INRIX, 2017), which makes the 120 outstanding requests for proposals for city and institutional parking management systems not surprising (IPMI, 2019).</p>
<p>The original concept for LaRC Park, proposed by a NASA employee, was to place sensors in each parking space. While this may have provided the most accurate results, we realized that due to exposure to weather, security, gaps in connectivity, etc., the method was impractical. We then looked at different systems, including Light Detection and Ranging (LIDAR) and image recognition.</p>
<p>Each had its own issues; LIDAR systems required power and connectivity where there was none, and most image recognition systems required an overhead angle of view.</p>
<p>The break came after reading Adam Geitgey awesome article, which introduced us to Matterport Mask-RCNN.</p>
<br>
<p>Jrei Dimanno, Justine Forrest, Rob Garcia, John Graham, Tanner Griffin, Ryan Hashi, Brandon Hutton, Fernando Jauregui, Wesley Madden, Doug Trent</p>
<br>
<p>Geitgey, A. (2019, January 21). Snagging parking spaces with Mask R-CNN and Python. Retrieved from <a href="https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400" target="_blank" title="">https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnnand-python-955f2231c400</a></p>
<p>INRIX. (2017, July 12). Searching for parking costs Americans $73 billion a year. Retrieved from <a href="http://inrix.com/press-releases/parking-pain-us/" target="_blank" title="">http://inrix.com/press-releases/parking-pain-us/</a></p>
<p>International Parking and Mobility Institute. (2015, October 12). Open requests for proposals. Retrieved July 31, 2019, from <a href="https://www.parking.org/membership/member-resources/rfps/" target="_blank" title="">https://www.parking.org/membership/member-resources/rfps/</a></p>
<p>Matterport, Inc. (2019, March 10). matterport/Mask_RCNN. Retrieved August 2, 2019, from <a href="https://github.com/matterport/Mask_RCNN" target="_blank" title="">https://github.com/matterport/Mask_RCNN</a></p>
